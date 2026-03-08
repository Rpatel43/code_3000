import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    common_cols = list(set(anon_df.columns) & set(aux_df.columns))
    id_cols = ["anon_id", "id", "patient_id", "record_id"]
    quasi_ids = [col for col in common_cols if col not in id_cols]

    if not quasi_ids:
        return pd.DataFrame(columns=["anon_id", "matched_name"])
    
    merged = anon_df.merge(aux_df, on=quasi_ids, how="inner", suffixes = ("_anon", "_aux"))

    match_counts = merged.groupby("anon_id").size().reset_index(name="match_count")

    unique_matches = match_counts[match_counts["match_count"] == 1]

    result = merged[merged["anon_id"].isin(unique_matches["anon_id"])]


    name_col = None
    for possible_name in ["name", "full_name", "patient_name", "Name"]:
        if possible_name in result.columns:
            name_col = possible_name
            break 

    if name_col:
        result = result[["anon_id", name_col]].rename(columns={name_col: "matched_name"})
    else:
        result = pd.DataFrame({"anon_id" : unique_matches["anon_id"], "matches_name" : ["Unknown"] * len(unique_matches)})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(matches_df) == 0:
        return 0.0

    unique_matched = matches_df["anon_id"].nunique()

    rate = unique_matched / len(anon_df)

    return rate