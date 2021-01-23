import datetime
import arrow

from typing import List


def is_leap_year(years: int):
    """
    if year is a leap year
    """

    assert isinstance(years, int), "Integer required."

    if ((years % 4 == 0 and years % 100 != 0) or (years % 400 == 0)):
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
