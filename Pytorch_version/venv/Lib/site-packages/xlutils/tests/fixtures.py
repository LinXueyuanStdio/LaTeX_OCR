# Copyright (c) 2008 Simplistix Ltd
#
# This Software is released under the MIT License:
# http://www.opensource.org/licenses/mit-license.html
# See license.txt for more details.

import sys
import os.path

from xlrd import XL_CELL_TEXT,Book
from xlrd.biffh import FUN 
from xlrd.formatting import XF, Format, Font, XFAlignment, XFBorder, XFBackground, XFProtection

from xlrd.sheet import Sheet

test_files = os.path.dirname(__file__)

test_xls_path = os.path.join(test_files,'test.xls')

class DummyBook(Book):

    biff_version = 80
    logfile = sys.stdout
    verbosity = 0
    datemode = 0
    on_demand = False

    def __init__(self,
                 formatting_info=0,
                 ragged_rows=False,
                 ):
        Book.__init__(self)
        self.ragged_rows = ragged_rows
        self.formatting_info=formatting_info
        self.initialise_format_info()
        if formatting_info:
            f = Font()
            self.font_list.append(f)
            self.format_map[0]= Format(0,FUN,u'General')
            xf = XF()
            xf.alignment = XFAlignment()
            xf.border = XFBorder()
            xf.background = XFBackground()
            xf.protection = XFProtection()
            self.xf_list.append(xf)
        
    def add(self,sheet):
        self._sheet_names.append(sheet.name)
        self._sheet_list.append(sheet)
        self.nsheets = len(self._sheet_list)

def make_book(rows=[]):
    book = DummyBook()
    sheet = make_sheet(rows,book=book)
    return book

def make_sheet(rows=(),book=None,name='test sheet',number=0):
    if book is None:
        book = DummyBook()
    book._sheet_visibility.append(0)
    sheet = Sheet(book,0,name,number)
    
    book.add(sheet)
    for rowx in range(len(rows)):
        row = rows[rowx]
        for colx in range(len(row)):
            value = row[colx]
            if isinstance(value,tuple):
                cell_type,value = value
            else:
                cell_type=XL_CELL_TEXT
            sheet.put_cell(rowx,colx,cell_type,value,0)
    sheet.tidy_dimensions()
    return sheet
