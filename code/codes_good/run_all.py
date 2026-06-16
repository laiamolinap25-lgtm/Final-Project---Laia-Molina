
import os
from config import biomarkers, neurotransmitters
from analysis import run_analysis


# Configuration


results_file = "all_correlations.csv"

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

generate_heatmaps = True



def clean_previous_results() -> None:

    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"Previous '{results_file}' removed.\n")


# Run full analysis

def run_all_analyses() -> None:

    total_jobs = (
        len(biomarker_list)
        * len(correlation_types)
        * len(neurotransmitter_selection)
    )

    current_job = 1

    for biomarker in biomarker_list:

        cfg = biomarkers[biomarker]
        output_dir = f"output/{biomarker}"

        print("\n" + "=" * 72)
        print(f"Biomarker: {biomarker}")
        print("=" * 72)

        for corr_type in correlation_types:

            print(f"\n  Correlation type: {corr_type}")

            for neuro_name in neurotransmitter_selection:

                print(
                    f"    [{current_job}/{total_jobs}] "
                    f"{neuro_name}"
                )

                run_analysis(
                    biomarker_name=biomarker,
                    biomarker_cfg=cfg,
                    neurotransmitter_name=neuro_name,
                    receptor_cols=neurotransmitters[neuro_name],
                    correlation_type=corr_type,
                    output_dir=output_dir,
                )

                current_job += 1



def run_heatmaps() -> None:

    if not generate_heatmaps:
        return

    print("\n" + "=" * 72)
    print("Generating heatmaps + LLM interpretations")
    print("=" * 72 + "\n")

    from plot_heatmap import generate_heatmaps_with_llm

    generate_heatmaps_with_llm(results_file)



def main() -> None:

    print("\n" + "=" * 72)
    print("Full neurotransmitter × ADNI pipeline")
    print("=" * 72 + "\n")

    clean_previous_results()

    run_all_analyses()

    run_heatmaps()

    print("\n" + "=" * 72)
    print("Pipeline finished successfully")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()