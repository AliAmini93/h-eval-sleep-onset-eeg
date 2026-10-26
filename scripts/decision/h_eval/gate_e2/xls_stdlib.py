"""Read ordinary BIFF8 XLS cells via Python standard library. Read-only, bounded.

Supports OLE2 FAT/miniFAT, BIFF8 SST/LABELSST, NUMBER/RK/MULRK, LABEL, BOOLERR.
Raises rather than inventing values on unsupported/corrupted workbook shapes.
"""
import struct
from pathlib import Path

FREE=0xffffffff; END=0xfffffffe; FAT=0xfffffffd; DIF=0xfffffffc

def u16(b,ofs=0):return struct.unpack_from('<H',b,ofs)[0]
def u32(b,ofs=0):return struct.unpack_from('<I',b,ofs)[0]
def u64(b,ofs=0):return struct.unpack_from('<Q',b,ofs)[0]

class Ole:
    def __init__(self,path):
        self.b=Path(path).read_bytes()
        if self.b[:8]!=bytes.fromhex('d0cf11e0a1b11ae1') or len(self.b)<512:raise ValueError('Not OLE2 legacy XLS')
        self.secsize=1<<u16(self.b,30);self.minsize=1<<u16(self.b,32)
        if self.secsize not in (512,4096) or self.minsize!=64:raise ValueError('Unrecognized OLE sectors')
        self.cutoff=u32(self.b,56)
        difat=[u32(self.b,76+i*4) for i in range(109)]
        ds=u32(self.b,68)
        for _ in range(u32(self.b,72)):
            if ds in (FREE,END):raise ValueError('Broken DIFAT chain')
            chunk=self.sector(ds)
            difat.extend(u32(chunk,k*4) for k in range(self.secsize//4-1))
            ds=u32(chunk,self.secsize-4)
        ns=u32(self.b,44)
        ids=[x for x in difat if x!=FREE][:ns]
        if len(ids)!=ns:raise ValueError('OLE FAT count invalid')
        self.fat=[]
        for sn in ids:
            chunk=self.sector(sn)
            self.fat.extend(u32(chunk,k) for k in range(0,len(chunk),4))
        d=self.chain(u32(self.b,48),self.fat,65536)
        self.entries=[]
        for k in range(0,len(d)-127,128):
            entry=d[k:k+128]
            name_size=u16(entry,64)
            if name_size<2 or name_size>64:continue
            name=entry[:name_size-2].decode('utf-16le','replace')
            self.entries.append((name,entry[66],u32(entry,116),u64(entry,120)))
        roots=[e for e in self.entries if e[1]==5]
        if len(roots)!=1:raise ValueError('OLE root missing')
        self.root=self.chain(roots[0][2],self.fat,65536)[:roots[0][3]] if roots[0][3] else b''
        mf=u32(self.b,60);mn=u32(self.b,64)
        raw=self.chain(mf,self.fat,mn) if mn else b''
        self.minifat=[u32(raw,k) for k in range(0,len(raw),4)]
    def sector(self,sn):
        pos=512+sn*self.secsize
        if sn>len(self.b)//self.secsize or pos+self.secsize>len(self.b):raise ValueError('OLE sector out of bounds')
        return self.b[pos:pos+self.secsize]
    def chain(self,start,fat,limit):
        o=[];seen=set();n=start
        while n!=END:
            if n in (FREE,FAT,DIF) or n in seen or n>=len(fat) or len(o)>=limit:raise ValueError('Malformed OLE sector chain')
            seen.add(n);o.append(self.sector(n));n=fat[n]
        return b''.join(o)
    def stream(self,name):
        matches=[e for e in self.entries if e[1]==2 and e[0] in (name,name.capitalize())]
        if not matches:raise ValueError('No Workbook stream')
        _,_,start,size=matches[0]
        if size>=self.cutoff:return self.chain(start,self.fat,65536)[:size]
        bits=[];seen=set();n=start
        while n!=END:
            if n in seen or n>=len(self.minifat):raise ValueError('Invalid mini FAT')
            seen.add(n);p=n*self.minsize
            if p+self.minsize>len(self.root):raise ValueError('Mini stream beyond root')
            bits.append(self.root[p:p+self.minsize]);n=self.minifat[n]
        return b''.join(bits)[:size]

def rk(v):
    div100=bool(v&1);is_int=bool(v&2)
    if is_int:
        z=struct.unpack('<i',struct.pack('<I',v&0xfffffffc))[0]>>2
    else:
        z=struct.unpack('<d',struct.pack('<Q',(v&0xfffffffc)<<32))[0]
    return z/100 if div100 else z

def unicode_string(b,pos):
    if len(b)<pos+3:raise ValueError('Short BIFF SST string')
    nchars=u16(b,pos);opts=b[pos+2];pos+=3
    nruns=0;exbytes=0
    if opts&8:nruns=u16(b,pos);pos+=2
    if opts&4:exbytes=u32(b,pos);pos+=4
    size=nchars*(2 if opts&1 else 1)
    if pos+size+nruns*4+exbytes>len(b):raise ValueError('Unsupported or broken BIFF SST continuation')
    value=b[pos:pos+size].decode('utf-16le' if opts&1 else 'latin1','replace')
    return value,pos+size+nruns*4+exbytes

def read_sheets(path):
    b=Ole(path).stream('Workbook')
    if len(b)>50*1024*1024:raise ValueError('Oversized XLS workbook stream')
    pos=0; bounds=[];sst=[];records=[]
    while pos+4<=len(b):
        typ=u16(b,pos);n=u16(b,pos+2);p=pos+4
        if p+n>len(b):raise ValueError('Truncated BIFF record')
        data=b[p:p+n];records.append((pos,typ,data));pos=p+n
        if typ==0x0085: # BOUNDSHEET
            offset=u32(data,0);cnt=data[6];unicodeflag=data[7]&1
            name=data[8:8+cnt*(2 if unicodeflag else 1)].decode('utf-16le' if unicodeflag else 'latin1')
            bounds.append((name,offset))
    # Concatenate SST plus continuation data; bail out if complex boundaries destroy strings.
    for i,(ofs,typ,data) in enumerate(records):
        if typ==0x00fc:
            payload=bytearray(data)
            j=i+1
            while j<len(records) and records[j][1]==0x003c:
                payload.extend(records[j][2]);j+=1
            if len(payload)<8:raise ValueError('Broken SST')
            count=u32(payload,4);p=8
            if count>100000:raise ValueError('SST too large')
            for _ in range(count):
                v,p=unicode_string(payload,p);sst.append(v)
            break
    if not bounds:raise ValueError('No BIFF worksheets')
    out={}
    for name,start in bounds:
        idx=next((i for i,r in enumerate(records) if r[0]==start),None)
        if idx is None:raise ValueError('BIFF sheet address absent')
        grid={};mxr=mxc=0
        for _,typ,data in records[idx:]:
            if typ==0x000a:break
            if typ in (0x00fd,0x0203,0x027e,0x0204,0x0006,0x0205,0x00bd):
                if len(data)<6:continue
                row=u16(data,0);col=u16(data,2)
                if row>30000 or col>255:raise ValueError('XLS sheet exceeds safety bounds')
                val=None
                if typ==0x00fd:
                    n=u32(data,6)
                    if n>=len(sst):raise ValueError('SST index out of bounds')
                    val=sst[n]
                elif typ==0x0203:val=struct.unpack_from('<d',data,6)[0]
                elif typ==0x027e:val=rk(u32(data,6))
                elif typ==0x0204:
                    sz=u16(data,6)
                    val=data[8:8+sz].decode('latin1','replace')
                elif typ==0x0006:
                    # Formula cached numerical result only; refuse error-valued formulas.
                    if data[12:14]!=b'\xff\xff':val=struct.unpack_from('<d',data,6)[0]
                elif typ==0x0205:val=bool(data[6]) if data[7]==0 else None
                elif typ==0x00bd:
                    last=u16(data,len(data)-2)
                    for k in range(last-col+1):
                        off=4+6*k
                        if off+6>len(data)-2:raise ValueError('Bad MULRK record')
                        grid[(row,col+k)]=rk(u32(data,off+2))
                    mxc=max(mxc,last)
                if val is not None:grid[(row,col)]=val
                mxr=max(mxr,row);mxc=max(mxc,col)
        if mxr>=3000 or mxc>=100:raise ValueError('Metadata XLS unexpectedly large')
        out[name]=[[grid.get((r,c),'') for c in range(mxc+1)] for r in range(mxr+1)]
    return out
