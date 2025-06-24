TASK_DEFINITIONS = {
    "Task012": {
        "Task": "Histopathology sample type",
        "Type": "Classification",
        "Description": (
            "This task involves extracting the origin of the material described in the pathology report. "
            "The output should classify the tissue origin into one of the following categories: lung, lymph node, "
            "bronchus, liver, brain, bone, or other. The origin of the tissue is generally mentioned at the beginning "
            "of the report as aard materiaal."
        ),
        "Parser_Format": {
            "single_label_multi_class_classification": {
                "type": "str",
                "description": (
                    "The origin of the material. One of 'lung', 'lymph node', 'bronchus', 'liver', "
                    "'brain', 'bone', or 'other'."
                ),
                "literals": [
                    "lung",
                    "lymph node",
                    "bronchus",
                    "liver",
                    "brain",
                    "bone",
                    "other",
                ],
            }
        },
    },
    "Task013": {
        "Task": "Pulmonary nodule presence",
        "Type": "Classification",
        "Description": (
            "This task requires analyzing the text of radiology reports to identify whether a pulmonary nodule is "
            "specifically mentioned. It is only relevant whether one is written literally in the text or not. "
            "You should not make inferences of the patient's health based on the report. The result should be a binary "
            "classification: 1.0 if a nodule is mentioned, and 0.0 if it is not."
        ),
        "Parser_Format": {
            "single_label_binary_classification": {
                "type": "float",
                "description": (
                    "The final classification. 1.0 if a pulmonary nodule is mentioned in the text, and 0.0 if it is not."
                ),
            }
        },
    },
    "Task014": {
        "Task": "Kidney Abnormality Presence",
        "Type": "Classification",
        "Description": (
            "This task involves determining whether a radiology report mentions any abnormalities related to the kidneys. "
            "Abnormalities include renal cell carcinoma, angiomyolipoma, cysts, kidney stones, conjoined kidneys, cases "
            "with partial or full nephrectomy, and several other rare abnormalities. The output should be a binary "
            "classification: 1.0 if a kidney abnormality is mentioned, and 0.0 if it is not."
        ),
        "Parser_Format": {
            "single_label_binary_classification": {
                "type": "float",
                "description": (
                    "The final classification. 1.0 if a kidney abnormality is mentioned in the text, and 0.0 if it is not."
                ),
            }
        },
    },
    "Task015": {
        "Task": "Hip Kellgren-Lawrence scoring",
        "Type": "Classification",
        "Description": (
            "This task involves classifying the Kellgren-Lawrence grade of osteoarthritis for both the left and right sides "
            "as described in the radiology report. The grades range from 0 to 4, with additional categories for 'not applicable (n)' "
            "and 'prosthesis (p)'. The output should provide a classification for each side."
        ),
        "Parser_Format": {
            "left": {
                "type": "str",
                "description": (
                    "The Kellgren-Lawrence grade of osteoarthritis for the left hip. An integer from 0 to 4, "
                    "or one of 'n' for not applicable or 'p' for prosthesis."
                ),
                "literals": ["0", "1", "2", "3", "4", "n", "p"],
            },
            "right": {
                "type": "str",
                "description": (
                    "The Kellgren-Lawrence grade of osteoarthritis for the right hip. An integer from 0 to 4, "
                    "or one of 'n' for not applicable or 'p' for prosthesis."
                ),
                "literals": ["0", "1", "2", "3", "4", "n", "p"],
            },
        },
    },
    "Task016": {
        "Task": "Colon Histopathology Diagnosis",
        "Type": "Classification",
        "Description": (
            "For the given numeral, predict whether the specimen was obtained from 1) biopsy (true) or polypectomy (false), "
            "and whether the pathologist rated the specimen as 2) hyperplastic polyps, 3) low-grade dysplasia, 4) high-grade dysplasia, "
            "5) cancer, 6) serrated polyps, or 7) non-informative. Give a true or false answer for each of the categories."
        ),
        "Parser_Format": {
            "biopsy": {
                "type": "bool",
                "description": "True if the specimen was obtained from a biopsy, false if from a polypectomy.",
            },
            "hyperplastic_polyps": {
                "type": "bool",
                "description": "True if rated as hyperplastic polyps.",
            },
            "low_grade_dysplasia": {
                "type": "bool",
                "description": "True if rated as low-grade dysplasia.",
            },
            "high_grade_dysplasia": {
                "type": "bool",
                "description": "True if rated as high-grade dysplasia.",
            },
            "cancer": {"type": "bool", "description": "True if rated as cancer."},
            "serrated_polyps": {
                "type": "bool",
                "description": "True if rated as serrated polyps.",
            },
            "non_informative": {
                "type": "bool",
                "description": "True if the specimen is non-informative.",
            },
        },
    },
    "Task017": {
        "Task": "Predicting lesion size measurements",
        "Type": "Regression",
        "Description": (
            "You will be provided with a radiology report and instructed to extract one of the following lesion size measurements, "
            "depending on the specific task stated at the beginning of each case: (1) PDAC size, (2) pulmonary nodule size, or (3) RECIST target lesion sizes. "
            "For PDAC or pulmonary nodule tasks, extract the lesion size in millimeters and assign it to lesion_1; set lesion_2 through lesion_5 to 0. "
            "For RECIST tasks, extract the size in millimeters for each described target lesion (up to 5), and assign them to lesion_1 through lesion_5 in order; "
            "set any remaining lesion values to 0 if fewer than 5 are described."
        ),
        "Parser_Format": {
            "lesion_1": {
                "type": "float",
                "description": (
                    "The estimated size of either the PDAC, the largest pulmonary nodule, or the first RECIST target lesion in mm."
                ),
            },
            "lesion_2": {
                "type": "float",
                "description": "The size of the second RECIST lesion in mm, or 0 if not described.",
            },
            "lesion_3": {
                "type": "float",
                "description": "The size of the third RECIST lesion in mm, or 0 if not described.",
            },
            "lesion_4": {
                "type": "float",
                "description": "The size of the fourth RECIST lesion in mm, or 0 if not described.",
            },
            "lesion_5": {
                "type": "float",
                "description": "The size of the fifth RECIST lesion in mm, or 0 if not described.",
            },
        },
    },
    "Task018": {
        "Task": "Predicting prostate volume, PSA, and PSA density",
        "Type": "Regression",
        "Description": (
            "The goal is to predict the prostate volume (in cm3), PSA level (in ng/ml) and PSA density (in ng/ml/ml) based on the radiology report. When a value is not explicitly stated, it should be denoted as 0.0. When dimensions are given (e.g., “3 x 4 x 5 mm”), we used the ellipsoid formula to calculate the volume (π*l*w*h/6), with l, w, and h the spatial dimensions in mm. When a range is given (e.g., “PSA: 4-5”), provide the average (i.e., 4.5)."
        ),
        "Parser_Format": {
            "prostate_volume": {
                "type": "float",
                "description": "The estimated prostate volume in cm3, or 0.0 if not described.",
            },
            "PSA_level": {
                "type": "float",
                "description": ("The PSA level in ng/ml, or 0.0 if not described."),
            },
            "PSA_density": {
                "type": "float",
                "description": (
                    "The PSA density in ng/ml/cm3, or 0.0 if not described."
                ),
            },
        },
    },
    "Task019": {
        "Task": "Anonymization",
        "Type": "Single Label Named Entity Recognition",
        "Description": (
            "Identify and classify sequences of tokens in the given text that qualify as Personally Identifiable Information (PII). "
            "Replace each identified sequence with the corresponding predefined category tag. "
            "Categories include: <DATUM>, <PERSOON>, <RAPPORT_ID>, <PLAATS>, <PHINUMMER>, <STUDIE_NAAM>, "
            "<ACCREDITATIE_NUMMER>, <TIJD>, and <LEEFTIJD>. Use tags exactly as specified, and avoid tagging non-PII data like medical measurements."
        ),
        "Parser_Format": {
            "tagged_text": {
                "type": "str",
                "description": (
                    "The text with the identified PII entities replaced by their respective tags."
                ),
            }
        },
    },
}
