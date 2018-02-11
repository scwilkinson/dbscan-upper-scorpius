import pandas as pd

from .settings import REL_PARALLAX_ERROR_FILTER, REL_PARALLAX_ERROR_LIMIT


def _gen_distance_column(df):
    return 1000/df['parallax']


def _gen_rel_parallax_error_column(df):
    return abs(df['parallax_error'] / df['parallax'])


def _gen_distance_error_column(df):
    return (1000 * df['parallax_error']) / df['parallax'].pow(2)


def _filter_rel_parallax_error(df):
    if REL_PARALLAX_ERROR_FILTER:
        print("Filtering by rel_parallax_error")
        print("Max rel_parallax_error: {0}".format(REL_PARALLAX_ERROR_LIMIT))

        df = df[df['rel_parallax_error'] < REL_PARALLAX_ERROR_LIMIT]

        print("Filtered Sample Size: {0}".format(len(df)))
    else:
        print("Not filtering by rel_parallax_error")

    return df


def load_csv(filename):
    df = pd.read_csv(filename)

    df['distance'] = _gen_distance_column(df)
    df['rel_parallax_error'] = _gen_rel_parallax_error_column(df)
    df['distance_error'] = _gen_distance_error_column(df)

    print("{0} Loaded!".format(filename))
    print("Sample Size: {0}".format(len(df)))

    return _filter_rel_parallax_error(df)
