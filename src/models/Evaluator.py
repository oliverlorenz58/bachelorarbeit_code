def metrics(counts):
    def n(*outcomes):
        return sum(counts.get(o, 0) for o in outcomes)

    tp = n('Warn_Returned_Canceled', 'Warn_Returned_NotCanceled')
    fp_wrongwarn = n('WrongWarn_Returned_Canceled', 'WrongWarn_Returned_NotCanceled')
    fp_nowarn = n('Warn_NotReturned_Canceled', 'Warn_NotReturned_NotCanceled')
    fn = n('NoWarn_Returned')
    tn = n('NoWarn_NotReturned')

    returns = tp + fp_wrongwarn + fn
    no_returns = fp_nowarn + tn
    warnings = tp + fp_wrongwarn + fp_nowarn
    total = returns + no_returns

    right_cancel = n('Warn_Returned_Canceled')
    false_cancel = n('Warn_NotReturned_Canceled')
    prevented = right_cancel + n('WrongWarn_Returned_Canceled')

    precision = _div(tp, warnings)
    recall = _div(tp, returns)
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': _div(2 * precision * recall, precision + recall),
        'FPR': _div(fp_nowarn, no_returns),
        'RCR': _div(right_cancel, tp),
        'FCR': _div(false_cancel, no_returns),
        'VR': _div(prevented, returns),
        'WR': _div(warnings, total),
        'WRRA': _div(_div(warnings, total), _div(returns, total)),
    }


def _div(a, b):
    return a / b if b else 0.0
