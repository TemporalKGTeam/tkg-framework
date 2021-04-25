import arrow

from typing import List


def is_leap_year(years: int):
    """
    if year is a leap year
    """

    assert isinstance(years, int), "Integer required."

    if (years % 4 == 0 and years % 100 != 0) or (years % 400 == 0):
        days_sum = 366
        return days_sum
    else:
        days_sum = 365
        return days_sum


def get_all_days_of_year(years: int, format: str = "YYYY-MM-DD") -> List[str]:
    """
    get all days of the year in string format
    """

    start_date = '%s-1-1' % years
    a = 0
    all_date_list = []
    days_sum = is_leap_year(int(years))
    while a < days_sum:
        b = arrow.get(start_date).shift(days=a).format(format)
        a += 1
        all_date_list.append(b)

    return all_date_list


def get_tem_dict():
    return {
        '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
        '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18,
        '10m': 19, '11m': 20, '12m': 21,
        '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
    }


def get_mod_dict():
    """
    Maps temporal modifiers to ids. <occursUntil> and <occursSince> are used in the YAGO datasets and uccurUntil and
    uccurSince are used in the Wikidata datasets.
    """
    return {
        '<occursSince>': 0,
        '<occursUntil>': 1,
        'occurUntil': 2,
        'occurSince': 3
    }
