import re

# EP_2807_MADC_p6_kl_EMG_kon5kg.csv

def pattern_parser(filepath):

    name = filepath.split('/')[-1].replace('.csv', '')
    parts = name.split('_')


    # ^-start of string
    # p - matches p letter
    # \d+ - one or more digits
    # $-end of string
    patient = next((p for p in parts if re.match(r"^p\d+$", p)), "unknown")

    exercise = parts[-1]

    limb_meaning = {"kl": "left limb", "kp": "right limb"}
    limb = next((limb_meaning[p] for p in parts if p in limb_meaning), "unknown")

    signal_type = "EMG" if "EMG" in parts else "unknown"

    return {
        "filename": name,
        "patient": patient,
        "limb": limb,
        "signal_type": signal_type,
        "exercise": exercise,
    }

