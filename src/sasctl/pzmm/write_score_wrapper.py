# Copyright (c) 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
- Collect inputs and outputs from inputVar.json and outputVar.json
    - Throw error or include optional arguments
- How to handle imports?
    - pass a list of imports, etc
- How do we handle model loading?
    - re.sub() with "Path(settings.pickle_path) / " and file name
    - Document replacement of path with inclusion of settings.pickle_path
- What form do we want to take the score snippet in?
    - function...we'll parse and drop in the contents
    - file, but tell us the score function...include other functions and pull out score function info
"""


class ScoreWrapper:
    score_wrapper: str = ""

    @classmethod
    def write_score_wrapper(
        cls,
    ):

"""
<imports provided by user>
import settings
<model load>
    add Path(settings.pickle_path
function definition
docstring: output string
globalize model and try/except block
<model scoring code>
clean up output
"""