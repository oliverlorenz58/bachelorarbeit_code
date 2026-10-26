import random

import numpy as np


def gestaltung(retourenkosten=1.0, fehlalarmkosten=0.0, skalierung=1.0):
    r, f = retourenkosten, fehlalarmkosten
    return {outcome: skalierung * wert + 0.0 for outcome, wert in {   # + 0.0: keine -0.0
        'NoWarn_NotReturned': 1.0,
        'NoWarn_Returned': -r,
        'Warn_Returned_Canceled': 0.5,
        'Warn_Returned_NotCanceled': -r,
        'Warn_NotReturned_Canceled': -1.5 - f,
        'Warn_NotReturned_NotCanceled': -f,
    }.items()}


REWARDS = {                                              # p*
    'basis': gestaltung(),                                # 0.63
    'fehlalarm': gestaltung(fehlalarmkosten=0.5),         # 0.71
    'retoure_3': gestaltung(retourenkosten=3.0),          # 0.43
    'retoure_2': gestaltung(retourenkosten=2.0),          # 0.51
    'retoure_1.5': gestaltung(retourenkosten=1.5),        # 0.57
    'basis_x2': gestaltung(skalierung=2.0),               # 0.63
    'fehlalarm_0.25': gestaltung(fehlalarmkosten=0.25),   # 0.67
    'retoure_0.58': gestaltung(retourenkosten=7 / 12),    # 0.71
    'fehlalarm_1': gestaltung(fehlalarmkosten=1.0),       # 0.75
    'fehlalarm_2': gestaltung(fehlalarmkosten=2.0),       # 0.82
}


def schwelle(rewards, p_true_cancel=0.5, p_false_cancel=0.2):
    w = REWARDS[rewards] if isinstance(rewards, str) else rewards
    warnung_ohne = p_false_cancel * w['Warn_NotReturned_Canceled'] + (1 - p_false_cancel) * w['Warn_NotReturned_NotCanceled']
    warnung_mit = p_true_cancel * w['Warn_Returned_Canceled'] + (1 - p_true_cancel) * w['Warn_Returned_NotCanceled']
    verlust, gewinn = w['NoWarn_NotReturned'] - warnung_ohne, warnung_mit - w['NoWarn_Returned']
    return verlust / (verlust + gewinn)


class Stochastisch_Binär:
    def __init__(self, p_true_cancel, p_false_cancel, rewards='basis'):
        self.p_true_cancel = p_true_cancel
        self.p_false_cancel = p_false_cancel
        self.rewards = REWARDS[rewards]

    def evaluate(self, truth, action, reason=None):
        returned = truth == 1
        if action == 0:
            outcome = 'NoWarn_Returned' if returned else 'NoWarn_NotReturned'
        elif action == 1:
            p_cancel = self.p_true_cancel if returned else self.p_false_cancel
            outcome = ('Warn_Returned' if returned else 'Warn_NotReturned') + _cancel(p_cancel)
        else:
            raise ValueError(f"Ungültige Aktion {action}, erwartet 0 oder 1")
        return self.rewards[outcome], outcome


class Stochastisch_13:
    def __init__(self, p_true_cancel, p_wrong_reason_cancel, p_false_cancel, rewards='basis'):
        self.p_true_cancel = p_true_cancel
        self.p_wrong_reason_cancel = p_wrong_reason_cancel
        self.p_false_cancel = p_false_cancel
        self.rewards = {
            **REWARDS[rewards],
            'WrongWarn_Returned_Canceled': REWARDS[rewards]['Warn_Returned_Canceled'],
            'WrongWarn_Returned_NotCanceled': REWARDS[rewards]['Warn_Returned_NotCanceled'],
        }

    def evaluate(self, truth, action, reason):
        if not 0 <= action <= len(reason):
            raise ValueError(f"Ungültige Aktion {action}, erwartet 0 bis {len(reason)}")
        returned = truth == 1
        if action == 0:
            outcome = 'NoWarn_Returned' if returned else 'NoWarn_NotReturned'
        elif not returned:
            outcome = 'Warn_NotReturned' + _cancel(self.p_false_cancel)
        elif action == _draw_reason(reason):
            outcome = 'Warn_Returned' + _cancel(self.p_true_cancel)
        else:
            outcome = 'WrongWarn_Returned' + _cancel(self.p_wrong_reason_cancel)
        return self.rewards[outcome], outcome


def _cancel(p):
    return '_Canceled' if np.random.rand() < p else '_NotCanceled'


def _draw_reason(reason):
    if sum(reason) <= 0:
        return -1
    return random.choices(range(1, len(reason) + 1), weights=reason)[0]
