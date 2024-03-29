# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from __future__ import absolute_import, division, print_function

import csv

import datasets

_TRAIN_DOWNLOAD_URL = "train_yelp_food_classifier.txt"
_TEST_DOWNLOAD_URL = "test_yelp_food_classifier.txt"
_VALID_DOWNLOAD_URL = "valid_yelp_food_classifier.txt"


class YelpF(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["mexican","america","asian"]),
                }
            ),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        train_path = _TRAIN_DOWNLOAD_URL
        test_path = _TEST_DOWNLOAD_URL
        valid_path = _VALID_DOWNLOAD_URL
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            for id_, row in enumerate(csv_file):
                row = row.split("\t")
                label, description = row[0], row[1]
                #mapping string to label
                if label== "mexican":
                    label = 0
                elif label== "american":
                    label = 1
                elif label== "asian":
                    label = 2
                else:
                    continue

                text = description
                yield id_, {"text": text, "label": label}
