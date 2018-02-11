import click

from modules.data import load_csv
from modules.DBSCAN import cluster
from modules.k_distance import choose_eps


@click.command()
@click.option('--input')
@click.option('--output', default=None)
def run_DBSCAN(input, output):

    df = load_csv(input)

    parameters = {'eps': choose_eps(df), 'min_pts': 15}
    labelled_df = cluster(df, parameters)

    if output is not None:
        labelled_df.to_csv(output)


if __name__ == "__main__":
    run_DBSCAN()
