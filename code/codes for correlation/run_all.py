import os

from config import biomarkers, neurotransmitters
from analysis import run_full_pipeline, results_file


correlation_types = [
    "delta_all",
    "delta_declining",
    "baseline_all",
    "baseline_declining",
]

neurotransmitter_selection = [
    "dopamine",
    "ach",
    "gaba",
    "serotonin",
    "glutamate",
]

biomarker_list = [
    "fdg",
    "t1t2",
    "volume",
    "thickness",
]


def clean_previous_results() -> None:
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"Previous '{results_file}' removed.\n")


def main() -> None:
    print("\n" + "=" * 72)
    print("Full neurotransmitter x ADNI pipeline")
    print("=" * 72 + "\n")

    clean_previous_results()

    run_full_pipeline(
        biomarkers=biomarkers,
        neurotransmitters=neurotransmitters,
        correlation_types=correlation_types,
        biomarker_list=biomarker_list,
        neurotransmitter_selection=neurotransmitter_selection,
    )

    print("\n" + "=" * 72)
    print("Pipeline finished successfully")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()