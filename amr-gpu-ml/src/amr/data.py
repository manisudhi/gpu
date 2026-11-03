"""
src/amr/data.py
Dataset ingestion, parsing, cleaning, and splitting module.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger("amr.data")

# Standard antibiotic class mapping for MDR definition (Magiorakos et al., 2012)
ANTIBIOTIC_CLASS_MAP = {
    # Carbapenems
    "meropenem": "Carbapenems",
    "imipenem": "Carbapenems",
    "ertapenem": "Carbapenems",
    "doripenem": "Carbapenems",
    # Cephalosporins (3rd/4th gen)
    "ceftazidime": "Cephalosporins",
    "cefepime": "Cephalosporins",
    "cefotaxime": "Cephalosporins",
    "ceftriaxone": "Cephalosporins",
    # Penicillins with beta-lactamase inhibitors
    "piperacillin-tazobactam": "BetaLactam_Inhibitor",
    "ticarcillin-clavulanic acid": "BetaLactam_Inhibitor",
    # Monobactams
    "aztreonam": "Monobactams",
    # Fluoroquinolones
    "ciprofloxacin": "Fluoroquinolones",
    "levofloxacin": "Fluoroquinolones",
    # Aminoglycosides
    "amikacin": "Aminoglycosides",
    "gentamicin": "Aminoglycosides",
    "tobramycin": "Aminoglycosides",
    # Polymyxins
    "colistin": "Polymyxins",
    "polymyxin B": "Polymyxins",
    # Folate pathway inhibitors
    "trimethoprim-sulfamethoxazole": "Folate_Inhibitors",
    # Tetracyclines / Glycylcyclines
    "doxycycline": "Tetracyclines",
    "minocycline": "Tetracyclines",
    "tigecycline": "Glycylcyclines",
}


def parse_amr_genotypes(raw_genotypes: str) -> List[str]:
    """
    Extract clean gene names and point mutations from NCBI AMR_genotypes string.
    Example: 'aac(6\')-Ib,blaKPC-2,gyrA_S83I=POINT,fosA' -> ['aac(6\')-Ib', 'blaKPC-2', 'gyrA_S83I', 'fosA']
    """
    if not isinstance(raw_genotypes, str) or not raw_genotypes or raw_genotypes == "NULL":
        return []
    
    genes = []
    # Strip quotes if present
    cleaned = raw_genotypes.strip('"\'')
    tokens = cleaned.split(",")
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Drop suffixes like =POINT, =COMPLETE, =PARTIAL, etc., but keep mutation identifier if part of gene
        if "=" in token:
            gene_name = token.split("=")[0].strip()
        else:
            gene_name = token
        if gene_name:
            genes.append(gene_name)
    return genes


def parse_ast_phenotypes(raw_ast: str) -> Dict[str, str]:
    """
    Parse AST_phenotypes string into a dictionary of antibiotic -> interpretation.
    Example: 'amikacin=I,cefepime=R,meropenem=S' -> {'amikacin': 'I', 'cefepime': 'R', 'meropenem': 'S'}
    """
    if not isinstance(raw_ast, str) or not raw_ast or raw_ast == "NULL":
        return {}
    
    ast_dict = {}
    cleaned = raw_ast.strip('"\'')
    pairs = cleaned.split(",")
    for pair in pairs:
        pair = pair.strip()
        if "=" in pair:
            parts = pair.split("=")
            drug = parts[0].strip().lower()
            val = parts[1].strip().upper()
            if val in ("S", "I", "R"):
                ast_dict[drug] = val
    return ast_dict


def parse_collection_year(date_str: str) -> Optional[int]:
    """Extract 4-digit collection year from various date string formats."""
    if not isinstance(date_str, str) or not date_str or date_str == "NULL":
        return None
    match = re.search(r"\b(19\d\d|20\d\d)\b", date_str)
    if match:
        year = int(match.group(1))
        if 1990 <= year <= 2026:
            return year
    return None


def clean_geo_location(geo_str: str) -> str:
    """Extract standard country name from geo_loc_name field."""
    if not isinstance(geo_str, str) or not geo_str or geo_str == "NULL":
        return "Unknown"
    # Often 'USA: California' or 'United Kingdom: London'
    country = geo_str.split(":")[0].strip()
    return country if country else "Unknown"


def clean_isolation_source(source_str: str) -> str:
    """Group isolation source into standardized categories."""
    if not isinstance(source_str, str) or not source_str or source_str == "NULL":
        return "Unknown"
    s = source_str.lower()
    if any(k in s for k in ["blood", "serum"]):
        return "Blood"
    elif any(k in s for k in ["urine", "urinary"]):
        return "Urine"
    elif any(k in s for k in ["sputum", "bronch", "lung", "respiratory", "tracheal"]):
        return "Respiratory"
    elif any(k in s for k in ["wound", "pus", "swab", "tissue", "skin", "abscess"]):
        return "Wound_Tissue"
    elif any(k in s for k in ["feces", "stool", "rectal", "gut", "cecum"]):
        return "Gastrointestinal"
    elif any(k in s for k in ["environmental", "ocean", "water", "sewage", "soil"]):
        return "Environmental"
    else:
        return "Other_Clinical"


def load_raw_dataset(tsv_path: Path) -> pd.DataFrame:
    """
    Load raw NCBI TSV file, extracting and standardizing target fields.
    """
    logger.info(f"Loading raw dataset from {tsv_path}")
    if not tsv_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    logger.info(f"Loaded {len(df):,} raw rows, {len(df.columns)} columns.")

    # Identify relevant columns
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip("#")
        if c_low in ["label", "isolate", "biosample_acc", "biosample"]:
            if "isolate_id" not in col_map:
                col_map["isolate_id"] = col
        elif c_low in ["collection_date", "create date", "create_date"]:
            if "date" not in col_map:
                col_map["date"] = col
        elif c_low in ["geo_loc_name", "location"]:
            if "location" not in col_map:
                col_map["location"] = col
        elif c_low in ["isolation_source"]:
            if "source" not in col_map:
                col_map["source"] = col
        elif c_low in ["host"]:
            if "host" not in col_map:
                col_map["host"] = col
        elif c_low in ["amr_genotypes", "amr genotypes"]:
            if "amr_genotypes" not in col_map:
                col_map["amr_genotypes"] = col
        elif c_low in ["ast_phenotypes"]:
            if "ast_phenotypes" not in col_map:
                col_map["ast_phenotypes"] = col

    logger.info(f"Mapped columns: {col_map}")

    # Build standardized dataframe
    clean_records = []
    has_ast = "ast_phenotypes" in col_map

    for idx, row in df.iterrows():
        iso_id = str(row[col_map["isolate_id"]]) if "isolate_id" in col_map else f"iso_{idx}"
        raw_date = str(row[col_map["date"]]) if "date" in col_map else ""
        raw_loc = str(row[col_map["location"]]) if "location" in col_map else ""
        raw_src = str(row[col_map["source"]]) if "source" in col_map else ""
        raw_host = str(row[col_map["host"]]) if "host" in col_map else ""
        raw_amr = str(row[col_map["amr_genotypes"]]) if "amr_genotypes" in col_map else ""
        raw_ast = str(row[col_map["ast_phenotypes"]]) if has_ast else ""

        genes = parse_amr_genotypes(raw_amr)
        ast_dict = parse_ast_phenotypes(raw_ast) if has_ast else {}
        year = parse_collection_year(raw_date)
        country = clean_geo_location(raw_loc)
        source = clean_isolation_source(raw_src)
        host = "Human" if "sapiens" in raw_host.lower() or "human" in raw_host.lower() else ("Animal" if raw_host and raw_host != "NULL" else "Unknown")

        clean_records.append({
            "isolate_id": iso_id,
            "year": year,
            "country": country,
            "source": source,
            "host": host,
            "genes": genes,
            "gene_count": len(genes),
            "ast_dict": ast_dict,
            "has_ast": len(ast_dict) > 0,
        })

    clean_df = pd.DataFrame(clean_records)
    logger.info(f"Processed into standardized dataframe: {len(clean_df):,} isolates, {clean_df['has_ast'].sum():,} with AST.")
    return clean_df


def create_time_aware_split(df: pd.DataFrame, split_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: train on historical isolates (<= split_year),
    test on future isolates (> split_year) to prevent isolate clonal leakage.
    """
    valid_years = df[df["year"].notna()]
    train_df = valid_years[valid_years["year"] <= split_year].copy()
    test_df = valid_years[valid_years["year"] > split_year].copy()
    logger.info(f"Time-aware split (cutoff {split_year}): Train={len(train_df):,}, Test={len(test_df):,}")
    return train_df, test_df
