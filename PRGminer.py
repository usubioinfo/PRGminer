#!/usr/bin/env python3

"""
PRGminer: Deep Learning-Based Plant Resistance Gene Prediction Tool
Copyright (C) 2023 Naveen Duhan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Naveen Duhan
Lab: KAABiL (Kaundal Artificial Intelligence & Advanced Bioinformatics Lab)
Version: 0.1
License: GPL-3.0
"""

import sys
import logging
from PRGminer.__main__ import main
from PRGminer import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_main():
    """
    Wrapper function to run the main PRGminer pipeline with error handling.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info(f"Starting PRGminer v{__version__}")
        return main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(run_main())