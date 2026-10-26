"""Scores, AUROC variants, signed stage attribution, person-cluster bootstrap, held-out rules, wake-to-N1 analysis."""
import numpy as np
import pytest

from heval.analysis import wake_to_n1 as w2n
from heval.evaluation import bootstrap, heldout, metrics, stage_analysis
from heval.models import references
from heval.synthetic import synthetic_nights

GOLD_CONTRAST = dict(delta_ww=0.12713684372030634, R=0.628330224700063, delta_se=0.13009511598103912,
                     delta_ww_ci95=[0.11215972682726483, 0.1422238265472668], R_ci95=[0.5525883232192881, 0.6949420955462716],
                     delta_se_ci95=[0.11208981568451187, 0.1469697454701902])


@pytest.fixture(scope='module')
def scored():
    """16 synthetic persons; reference = constant, candidate = TRAIN-fitted Stage+time reference."""
    d = synthetic_nights(n_persons=16, seed=2, with_raw=False)
    s1, y = (d['stage'] == '1').astype(float), d['y'].astype(float)
    B = references.fit_B(d['elapsed_s'], s1, y, (3, 10.0, True))
    return dict(y=y, person=d['person'], p_ref=np.full(len(y), y.mean()), p_c=references.predict_B(B, d['elapsed_s'], s1),
                stage=d['stage'], elapsed=d['elapsed_s'], recording=d['recording'])


# ------------------------------------------------------------------------ proper scores and calibration
def test_brier_logloss_calibration():
    p, y = np.array([0.1, 0.8, 0.5, 1.0]), np.array([0, 1, 1, 1])
    assert np.allclose(metrics.brier(p, y), [0.01, 0.04, 0.25, 0.0])
    assert np.allclose(metrics.logloss(p, y), [-np.log(0.9), -np.log(0.8), -np.log(0.5), -np.log(1 - 1e-6)])
    c = metrics.calibration(np.array([0.2, 0.2, 0.6, 0.6]), np.array([0, 0, 1, 0]))
    assert c['cal_in_large'] == pytest.approx(0.4 - 0.25) and c['ece10'] == pytest.approx((2 * 0.2 + 2 * 0.1) / 4)
    assert metrics.calibration_in_stratum(np.array([0.3, 0.4]), np.array([1, 1]))[2] == 'undefined: outcome constant in stratum'
    with pytest.raises(ValueError, match='probabilities'):
        metrics.brier(np.array([1.2]), np.array([1]))
    with pytest.raises(ValueError, match='0/1'):
        metrics.logloss(np.array([0.5]), np.array([2]))


# ------------------------------------------------------------------------ AUROC: ties, cross-person vs within-recording
def test_auroc_ties_and_cross_person_vs_within_recording():
    assert metrics.auroc(np.array([0.2, 0.5, 0.5, 0.9]), np.array([0, 0, 1, 1])) == (1 + 0.5 + 1 + 1) / 4
    assert metrics.auroc(np.array([0.3, 0.3]), np.array([0, 1])) == 0.5
    assert metrics.auroc(np.array([0.1, 0.2]), np.array([0, 0])) is None
    # person A runs high, person B low; inside each recording the ranking is perfect
    p = np.array([0.8, 0.9, 0.1, 0.2])
    y = np.array([0, 1, 0, 1])
    person = np.array(['A', 'A', 'B', 'B'])
    w = metrics.within_recording_auroc(p, y, person)
    assert w['mean'] == 1.0 and w['n_defined'] == 2
    m = metrics.matched_auroc(p, y, person, np.zeros(4, int), np.random.default_rng(0))
    assert m['n_pairs'] == 2 and m['auroc'] == 0.5 and m['ci95'] == 'not_reported_lt_8_persons'   # B+ vs A-: 0, A+ vs B-: 1
    assert metrics.auroc(p, y) == 0.75


# ------------------------------------------------------------------------ signed W / Stage-1 attribution
def test_signed_contribution_identity_negative_wake_and_n1_share_over_100():
    b_ref = np.array([0.30, 0.20, 0.10, 0.40, 0.50])
    b_cand = np.array([0.35, 0.25, 0.15, 0.10, 0.20])          # worse on wake windows, better on Stage 1
    cats = np.array(['W', 'W', 'W', '1', '1'])
    c = stage_analysis.contributions(b_ref, b_cand, cats)
    assert c['delta_ww_all'] == pytest.approx(np.mean(b_ref - b_cand))
    assert c['absolute']['W'] == pytest.approx(-0.15 / 5) and c['absolute']['1'] == pytest.approx(0.6 / 5)
    assert sum(c['absolute'].values()) == pytest.approx(c['delta_ww_all'], abs=1e-15)          # original denominator
    assert c['share']['1'] > 1 and c['share']['W'] < 0 and sum(c['share'].values()) == pytest.approx(1.0)
    assert stage_analysis.contributions(np.ones(2), np.ones(2), np.array(['W', '1']))['share'] == 'undefined_delta_near_0'
    with pytest.raises(ValueError, match='equal length'):
        stage_analysis.contributions(np.ones(3), np.ones(2), np.array(['W', '1', '1']))


def test_stage_attribution_summary(scored):
    s = scored
    out = stage_analysis.stage_attribution(s['y'], s['person'], s['recording'], s['stage'], s['elapsed'] < 300,
                                           {'C': s['p_ref'], 'B': s['p_c']}, {'C->B': ('C', 'B')})
    w, one = out['subsets']['STAGE_W']['contrasts']['C->B'], out['subsets']['STAGE_1']['contrasts']['C->B']
    contrib = out['stage_contribution']['C->B']['absolute']
    n = len(s['y'])
    assert contrib['W'] == pytest.approx(w['delta_ww'] * w['n_windows'] / n) and contrib['1'] == pytest.approx(one['delta_ww'] * one['n_windows'] / n)
    assert out['subsets']['ALL']['contrasts']['C->B']['delta_ww'] == pytest.approx(contrib['W'] + contrib['1'])


# ------------------------------------------------------------------------ person-cluster bootstrap
def test_bootstrap_golden_determinism_and_person_resampling(scored):
    s = scored
    I = bootstrap.boot_index(16, seed=20261026, n=2000)
    assert np.array_equal(I, bootstrap.boot_index(16, seed=20261026, n=2000))
    assert not np.array_equal(I, bootstrap.boot_index(16, seed=1, n=2000))
    out = bootstrap.reference_contrast(s['p_ref'], s['p_c'], s['y'], s['person'], I)
    for k, v in GOLD_CONTRAST.items():
        assert np.allclose(out[k], v, rtol=1e-10), k
    assert out == {k: v for k, v in bootstrap.reference_contrast(s['p_ref'], s['p_c'], s['y'], s['person'], I).items()}
    assert out['delta_ww_ci95'][0] <= out['delta_ww'] <= out['delta_ww_ci95'][1]
    assert out['R'] == pytest.approx(out['delta_ww'] / out['brier_ref_ww'])
    # replicate 0 by hand: whole persons are resampled and their windows' sums pooled (ratio estimator)
    pu = sorted(set(s['person']))
    br = np.array([((s['p_ref'] - s['y']) ** 2)[s['person'] == q].sum() for q in pu])
    bc = np.array([((s['p_c'] - s['y']) ** 2)[s['person'] == q].sum() for q in pu])
    nw = np.array([(s['person'] == q).sum() for q in pu])
    rep0 = (br[I[0]].sum() - bc[I[0]].sum()) / nw[I[0]].sum()
    reps = [(br[r].sum() - bc[r].sum()) / nw[r].sum() for r in I]
    assert np.percentile(reps, [2.5, 97.5]).tolist() == pytest.approx(out['delta_ww_ci95'], rel=1e-12) and np.isfinite(rep0)
    assert bootstrap.stratum_contrast(s['p_ref'], s['p_c'], s['y'], s['person'], I)['R_ci95'] == pytest.approx(GOLD_CONTRAST['R_ci95'])


def test_interval_suppressed_below_eight_persons(scored):
    s = scored
    m = np.isin(s['person'], sorted(set(s['person']))[:7])
    for f in (bootstrap.reference_contrast, bootstrap.stratum_contrast):
        out = f(s['p_ref'][m], s['p_c'][m], s['y'][m], s['person'][m], bootstrap.boot_index(7))
        assert out['ci'] == 'not_reported_lt_8_persons' and 'delta_ww_ci95' not in out and out['n_persons'] == 7
    m8 = np.isin(s['person'], sorted(set(s['person']))[:8])
    assert 'delta_ww_ci95' in bootstrap.reference_contrast(s['p_ref'][m8], s['p_c'][m8], s['y'][m8], s['person'][m8], bootstrap.boot_index(8, n=200))


def test_bootstrap_rejects_invalid_input(scored):
    s = scored
    with pytest.raises(ValueError, match='equal length'):
        bootstrap.reference_contrast(s['p_ref'][:-1], s['p_c'], s['y'], s['person'], None)
    with pytest.raises(ValueError, match='probabilities'):
        bootstrap.stratum_contrast(s['p_ref'] + 2, s['p_c'], s['y'], s['person'], None)
    with pytest.raises(ValueError, match='width'):
        bootstrap.stratum_contrast(s['p_ref'], s['p_c'], s['y'], s['person'], bootstrap.boot_index(5, n=10))


# ------------------------------------------------------------------------ prespecified held-out logic
def test_scorer_and_h1_decision(scored):
    s = scored
    rows = [dict(person=p, y=int(yy), elapsed_s=float(e)) for p, yy, e in zip(s['person'], s['y'], s['elapsed'])]
    S = heldout.Scorer(rows, {'CONSTANT': s['p_ref'], 'EEG_LOGIT': s['p_c']})
    c = S.contrast(S.brier(s['p_ref']), S.brier(s['p_c']))
    assert c['R'] == pytest.approx(0.628330224700063, rel=1e-12)
    assert np.allclose(c['R_ci95'], [0.5507835278191389, 0.699653246640148], rtol=1e-10)
    assert S.permutation_null('EEG_LOGIT')['null_mean'] == pytest.approx(0.49925957504034113, rel=1e-12)
    prim = dict(delta_ww_ci95=[0.01, 0.03], R=0.2, R_ci95=[0.06, 0.3], delta_ww=0.02)
    ok_nc2, bad_nc2 = dict(delta_ww_ci95=[-0.01, 0.01]), dict(delta_ww_ci95=[0.001, 0.01])
    assert heldout.decide(prim, 0.01, dict(ci95=[0.55, 0.7]), True, ok_nc2, True)['outcome'] == 'GO'
    assert heldout.decide(prim, 0.01, dict(ci95=[0.55, 0.7]), True, bad_nc2, True)['outcome'] == 'INCONCLUSIVE'   # donor reproduces gain
    assert heldout.decide(prim, 0.01, dict(ci95=[0.55, 0.7]), False, ok_nc2, True)['outcome'] == 'INVALID'
    nogo = dict(delta_ww_ci95=[-0.01, 0.01], R=0.01, R_ci95=[-0.1, 0.04], delta_ww=0.001)
    assert heldout.decide(nogo, 0.001, dict(ci95=[0.55, 0.7]), True, ok_nc2, True)['outcome'] == 'NO-GO'
    assert heldout.h2_outcome('GO', dict(R=0.1, delta_ww_ci95=[0.01, 0.1]), True) == 'SUPPORTED'
    assert heldout.h2_outcome('NO-GO', dict(R=0.1, delta_ww_ci95=[0.01, 0.1]), True) == 'NOT_TESTED(would_be_SUPPORTED)'
    assert heldout.h2_outcome('GO', dict(R=0.1, delta_ww_ci95=[0.01, 0.1]), False) == 'INVALID'
    plan, info = heldout.donor_plan(list(s['person']), list(s['elapsed']), 0)
    assert all(s['person'][j] != s['person'][i] for i, j in enumerate(plan)) and info['fallback_windows'] >= 0
    with pytest.raises(ValueError, match='aligned'):
        heldout.Scorer(rows, {'CONSTANT': s['p_ref'][:-1]})


# ------------------------------------------------------------------------ wake-to-first-N1 (TRAIN-only analysis)
def test_wake_to_n1_estimands_and_warnings():
    d = synthetic_nights(n_persons=4, seed=5, with_raw=False)
    e = w2n.estimands(d['stage'], d['elapsed_s'], d['first_sleep_s'], d['time_to_n2_s'], d['y'])
    assert np.array_equal(e['e1'], d['elapsed_s'] < d['first_sleep_s'])
    assert np.all(e['y1'][~e['e1']] == 0) and np.all(e['k1'][e['y1'] == 0] == 11) and np.all(e['k1'][e['y1'] == 1] <= 10)
    assert np.array_equal(e['e2'] | e['e3'], np.ones(len(d['y']), bool)) and not np.any(e['e2pre'] & e['e2ret'])
    # one recording, alarms at windows 2-3 (no event) and 6 (event) -> one false-alarm episode, event detected
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    p = np.array([0, 0, 1, 1, 0, 0, 1, 0], float)
    out = w2n.warnings(p, np.full(8, 0.5), y, np.array(['r'] * 8), np.arange(8) * 30., np.ones(8, bool))
    assert out['n_events'] == 1 and out['event_sensitivity'] == 1.0
    assert out['false_alarm_episodes_per_hour'] == pytest.approx(1 / (8 * 30 / 3600))


def test_wake_to_n1_paired_and_recalibration(scored):
    s = scored
    out = w2n.paired(s['p_ref'], s['p_c'], s['y'], s['person'])
    assert out['R'] == pytest.approx(0.628330224700063) and out['loo_R_min'] == pytest.approx(0.612020796437221, rel=1e-12)
    u = np.log1p(s['elapsed'] / 60)
    u = (u - u.mean()) / u.std()
    g = w2n.folds(s['person'], w2n.FOLD_SALTS['A'])
    _, info = w2n.calibrate_fold('K2', w2n.logit(s['p_c']), u, s['y'], g, w2n.logit(s['p_c'][:3]), u[:3])
    assert info['lambda_'] == 100.0 and info['a'] == pytest.approx(0.03417185722535213, abs=1e-6) and info['c'] == 0.0
    p1, _ = w2n.calibrate_fold('K1', w2n.logit(s['p_c']), u, s['y'], g, w2n.logit(s['p_c']), u)
    assert p1.mean() == pytest.approx(s['y'].mean(), abs=2e-3)               # intercept recalibration matches the base rate
    with pytest.raises(ValueError, match='K1, K2 or K3'):
        w2n.calibrate_fold('K9', None, None, None, None, None, None)
