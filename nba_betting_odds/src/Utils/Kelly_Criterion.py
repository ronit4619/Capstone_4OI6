def american_to_decimal(american_odds):
    if american_odds is None:
        raise ValueError("American odds cannot be None")
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet
    """
    decimal_odds = american_to_decimal(american_odds)
    bankroll_fraction = round((100 * (decimal_odds * model_prob - (1 - model_prob))) / decimal_odds, 2)
    return bankroll_fraction if bankroll_fraction > 0 else 0