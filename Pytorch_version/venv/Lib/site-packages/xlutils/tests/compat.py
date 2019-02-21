# Copyright (c) 2011-2013 Simplistix Ltd, 2015 Chris Withers
# See license.txt for license details.

# This module contains bits and pieces to achieve compatibility across all the
# versions of python supported.

import doctest
import re
import textwrap

import manuel
from manuel.codeblock import (
    CODEBLOCK_START,
    CODEBLOCK_END,
    CodeBlock,
    execute_code_block,
    )

from ..compat import PY3

BYTE_LITERALS = re.compile("b((:?'.*?')|(:?\".*?\"))", re.MULTILINE)
UNICODE_LITERALS = re.compile("u((:?'.*?')|(:?\".*?\"))", re.MULTILINE)


def version_agnostic(text):
    if PY3:
        regex = UNICODE_LITERALS
    else:
        regex = BYTE_LITERALS
    return regex.sub('\\1', text)


def find_code_blocks(document):
    for region in document.find_regions(CODEBLOCK_START, CODEBLOCK_END):
        start_end = CODEBLOCK_START.search(region.source).end()
        source = version_agnostic(textwrap.dedent(region.source[start_end:]))
        source = 'from __future__ import print_function\n' + source
        source_location = '%s:%d' % (document.location, region.lineno)
        code = compile(source, source_location, 'exec', 0, True)
        document.claim_region(region)
        region.parsed = CodeBlock(code, source)


class Manuel(manuel.Manuel):
    def __init__(self):
        manuel.Manuel.__init__(self, [find_code_blocks], [execute_code_block])


class DocTestChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        want = version_agnostic(want)
        return doctest.OutputChecker.check_output(
            self, want, got, optionflags
            )
