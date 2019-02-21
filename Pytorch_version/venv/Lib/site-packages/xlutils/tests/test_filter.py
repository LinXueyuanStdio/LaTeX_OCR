from __future__ import with_statement

# Copyright (c) 2008-2012 Simplistix Ltd
#
# This Software is released under the MIT License:
# http://www.opensource.org/licenses/mit-license.html
# See license.txt for more details.

from mock import Mock
from tempfile import TemporaryFile
from testfixtures import compare, Comparison as C, replace, log_capture, ShouldRaise, tempdir
from unittest import TestSuite,TestCase,makeSuite
from xlrd import open_workbook, XL_CELL_NUMBER, XL_CELL_ERROR, XL_CELL_BOOLEAN
from xlrd.formatting import XF
from xlutils.filter import BaseReader,GlobReader,MethodFilter,BaseWriter,process,XLRDReader,XLWTWriter, BaseFilter
from xlutils.tests.fixtures import test_files,test_xls_path,make_book,make_sheet,DummyBook
from ..compat import StringIO, PY3
from .compat import version_agnostic as va

import os

class TestReader(BaseReader):

    formatting_info = 0
    
    def __init__(self,*sheets,**books):
        self.setup(*sheets,**books)
        
    def setup(self,*sheets,**books):
        self.books = []
        if sheets:
            self.makeBook('test',sheets)
        for name,value in sorted(books.items()):
            self.makeBook(name,value)
        
    def makeBook(self,book_name,sheets):
        book = DummyBook(self.formatting_info)
        index = 0
        for name,rows in sheets:
            make_sheet(rows,book,name,index)
            index+=1
        self.books.append((book,book_name+'.xls'))
        
    def get_workbooks(self):
        return self.books

class TestBaseReader(TestCase):

    def test_no_implementation(self):
        r = BaseReader()
        f = Mock()
        with ShouldRaise(NotImplementedError()):
            r(f)
        self.assertEqual(f.method_calls,[('start',(),{})])
        
    def test_ragged_rows(self):
        class TestReader(BaseReader):
            def get_filepaths(self):
                return (os.path.join(test_files,'ragged.xls'),)
        t = TestReader()
        class TestFilter(BaseFilter):
            def cell(self,rdrowx,rdcolx,wtrowx,wtcolx):
                self.rdsheet.cell(rdrowx,rdcolx)
        f = TestFilter()
        f.next = Mock()
        t(f)
    
    def test_custom_filepaths(self):
        # also tests the __call__ method
        class TestReader(BaseReader):
            def get_filepaths(self):
                return (test_xls_path,)
        t = TestReader()
        f = Mock()
        t(f)
        compare(f.method_calls,[
            ('start',(),{}),
            ('workbook',(C('xlrd.Book',
                           formatting_info=1,
                           on_demand=True,
                           ragged_rows=True,
                           strict=False),'test.xls'),{}),
            ('sheet',(C('xlrd.sheet.Sheet'),u'Sheet1'),{}),
            ('row',(0,0),{}),
            ('cell',(0,0,0,0),{}),
            ('cell',(0,1,0,1),{}),
            ('row',(1,1),{}),
            ('cell',(1,0,1,0),{}),
            ('cell',(1,1,1,1),{}),
            ('sheet',(C('xlrd.sheet.Sheet'),u'Sheet2'),{}),
            ('row',(0,0),{}),
            ('cell',(0,0,0,0),{}),
            ('cell',(0,1,0,1),{}),
            ('row',(1,1),{}),
            ('cell',(1,0,1,0),{}),
            ('cell',(1,1,1,1),{}),
            ('finish',(),{}),
            ])

    def test_custom_getworkbooks(self):
        book = make_book((('1','2','3'),))
        class TestReader(BaseReader):
            def get_workbooks(self):
                yield book,'test.xls'
        t = TestReader()
        f = Mock()
        t(f)
        compare(f.method_calls,[
            ('start',(),{}),
            ('workbook',(C('xlutils.tests.fixtures.DummyBook'),'test.xls'),{}),
            ('sheet',(C('xlrd.sheet.Sheet'),'test sheet'),{}),
            ('row',(0,0),{}),
            ('cell',(0,0,0,0),{}),
            ('cell',(0,1,0,1),{}),
            ('cell',(0,2,0,2),{}),
            ('finish',(),{}),
            ])
        # check we're getting the right things
        self.failUnless(f.method_calls[1][1][0] is book)
        self.failUnless(f.method_calls[2][1][0] is book.sheet_by_index(0))
    
    class MockReader(BaseReader):
        def __init__(self,book):
            book.nsheets=2
            book.sheet_by_index.side_effect = self.sheet_by_index
            book.sheet0.nrows=1
            book.sheet0.ncols=1
            book.sheet0.row_len.return_value = 1
            book.sheet0.name='sheet0'
            book.sheet1.nrows=1
            book.sheet1.ncols=1
            book.sheet1.row_len.return_value = 1
            book.sheet1.name='sheet1'
            self.b = book
            
        def get_workbooks(self):
            return [(self.b,str(id(self.b)))]
        
        def sheet_by_index(self,i):
            return getattr(self.b,'sheet'+str(i))
    
    def test_on_demand_true(self):
        m = Mock()
        book = m.book
        book.on_demand=True
        r = self.MockReader(book)
        f = m.filter
        r(f)
        compare(m.method_calls,[
            ('filter.start', (), {}),
            ('filter.workbook', (book, str(id(book))), {}),
            ('book.sheet_by_index', (0,), {}),
            ('filter.sheet',(book.sheet0, 'sheet0'),{}),
            ('filter.row', (0, 0), {}),
            ('book.sheet0.row_len', (0,), {}),
            ('filter.cell', (0, 0, 0, 0), {}),
            ('book.unload_sheet', (0,), {}),
            ('book.sheet_by_index', (1,), {}),
            ('filter.sheet',(book.sheet1, 'sheet1'),{}),
            ('filter.row', (0, 0), {}),
            ('book.sheet1.row_len', (0,), {}),
            ('filter.cell', (0, 0, 0, 0), {}),
            ('book.unload_sheet', (1,), {}),
            ('filter.finish', (), {})
            ])
        
    def test_on_demand_false(self):
        m = Mock()
        book = m.book
        book.on_demand=False
        r = self.MockReader(book)
        f = m.filter
        r(f)
        compare(m.method_calls,[
            ('filter.start', (), {}),
            ('filter.workbook', (book, str(id(book))), {}),
            ('book.sheet_by_index', (0,), {}),
            ('filter.sheet',(book.sheet0, 'sheet0'),{}),
            ('filter.row', (0, 0), {}),
            ('book.sheet0.row_len', (0,), {}),
            ('filter.cell', (0, 0, 0, 0), {}),
            ('book.sheet_by_index', (1,), {}),
            ('filter.sheet',(book.sheet1, 'sheet1'),{}),
            ('filter.row', (0, 0), {}),
            ('book.sheet1.row_len', (0,), {}),
            ('filter.cell', (0, 0, 0, 0), {}),
            ('filter.finish', (), {})
            ])

class TestBaseFilter(TestCase):

    def setUp(self):
        from xlutils.filter import BaseFilter
        self.filter = BaseFilter()
        self.filter.next = self.tf = Mock()

    def test_start(self):
        self.filter.start()
        self.assertEqual(self.tf.method_calls,[
            ('start',(),{})
            ])
                         
    def test_workbook(self):
        self.filter.workbook('rdbook','wtbook_name')
        self.assertEqual(self.tf.method_calls,[
            ('workbook',('rdbook','wtbook_name'),{})
            ])
                         
    def test_sheet(self):
        self.filter.sheet('rdsheet','wtsheet_name')
        self.assertEqual(self.tf.method_calls,[
            ('sheet',('rdsheet','wtsheet_name'),{})
            ])
                         
    def test_set_rdsheet(self):
        self.filter.set_rdsheet('rdsheet2')
        self.assertEqual(self.tf.method_calls,[
            ('set_rdsheet',('rdsheet2',),{})
            ])
                         
    def test_row(self):
        self.filter.row(0,1)
        self.assertEqual(self.tf.method_calls,[
            ('row',(0,1),{})
            ])
                         
    def test_cell(self):
        self.filter.cell(0,1,2,3)
        self.assertEqual(self.tf.method_calls,[
            ('cell',(0,1,2,3),{})
            ])
                         
    def test_finish(self):
        self.filter.finish()
        self.assertEqual(self.tf.method_calls,[
            ('finish',(),{})
            ])

class OurMethodFilter(MethodFilter):
    def __init__(self,collector,call_on=True):
        MethodFilter.__init__(self,call_on)
        self.collector = collector        
    def method(self,name,*args):
        self.collector.append((name,args))
        
class TestMethodFilter(TestCase):

    def setUp(self):
        self.called = []

    def test_cmp(self):
        if PY3:
            MethodFilter() == OurMethodFilter([])
        else:
            cmp(MethodFilter(), OurMethodFilter([]))
        
    def do_calls_and_test(self,filter):
        filter.next = tf = Mock()
        filter.start()
        filter.workbook('rdbook','wtbook_name')
        filter.sheet('rdsheet','wtsheet_name')
        filter.row(0,1)
        filter.cell(0,1,2,3)
        filter.set_rdsheet('rdsheet2')
        filter.finish()
        self.assertEqual(tf.method_calls,[
            ('start',(),{}),
            ('workbook',('rdbook','wtbook_name'),{}),
            ('sheet',('rdsheet','wtsheet_name'),{}),
            ('row',(0,1),{}),
            ('cell',(0,1,2,3),{}),
            ('set_rdsheet',('rdsheet2',),{}),
            ('finish',(),{}),
            ])
        
    def test_all(self):
        self.do_calls_and_test(OurMethodFilter(self.called))
        compare(self.called,[
            ('start',()),
            ('workbook',('rdbook','wtbook_name')),
            ('sheet',('rdsheet','wtsheet_name')),
            ('row',(0,1)),
            ('cell',(0,1,2,3)),
            ('set_rdsheet',('rdsheet2',)),
            ('finish',()),
            ])

    def test_all_text(self):
        self.do_calls_and_test(OurMethodFilter(self.called,call_on='True'))
        compare(self.called,[
            ('start',()),
            ('workbook',('rdbook','wtbook_name')),
            ('sheet',('rdsheet','wtsheet_name')),
            ('row',(0,1)),
            ('cell',(0,1,2,3)),
            ('set_rdsheet',('rdsheet2',)),
            ('finish',()),
            ])

    def test_all_text_list(self):
        self.do_calls_and_test(OurMethodFilter(self.called,call_on=['True']))
        compare(self.called,[
            ('start',()),
            ('workbook',('rdbook','wtbook_name')),
            ('sheet',('rdsheet','wtsheet_name')),
            ('row',(0,1)),
            ('cell',(0,1,2,3)),
            ('set_rdsheet',('rdsheet2',)),
            ('finish',()),
            ])

    def test_somecalls_and_test(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['row','cell']))
        compare(self.called,[
            ('row',(0,1)),
            ('cell',(0,1,2,3)),
            ])

    def test_none(self):
        self.do_calls_and_test(OurMethodFilter(self.called,()))
        compare(self.called,[])

    def test_start(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['start']))
        compare(self.called,[
            ('start',()),
            ])

    def test_workbook(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['workbook']))
        compare(self.called,[
            ('workbook',('rdbook','wtbook_name')),
            ])

    def test_sheet(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['sheet']))
        compare(self.called,[
            ('sheet',('rdsheet','wtsheet_name')),
            ])

    def test_set_rdsheet(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['set_rdsheet']))
        compare(self.called,[
            ('set_rdsheet',('rdsheet2',)),
            ])

    def test_row(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['row']))
        compare(self.called,[
            ('row',(0,1)),
            ])

    def test_cell(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['cell']))
        compare(self.called,[
            ('cell',(0,1,2,3)),
            ])

    def test_finish(self):
        self.do_calls_and_test(OurMethodFilter(self.called,['finish']))
        compare(self.called,[
            ('finish',()),
            ])

    def test_invalid(self):
        with ShouldRaise(ValueError("'foo' is not a valid method name")):
            OurMethodFilter(self.called,['foo'])
        
    
from xlutils.filter import Echo

class TestEcho(TestCase):

    @replace('sys.stdout',StringIO())
    def test_method(self,out):
        filter = Echo(methods=['workbook'])
        filter.method('name','foo',1)
        compare(out.getvalue(),"name:('foo', 1)\n")
        
    @replace('sys.stdout',StringIO())
    def test_method_with_name(self,out):
        filter = Echo('echo',['workbook'])
        filter.method('name','foo',1)
        compare(out.getvalue(),"'echo' name:('foo', 1)\n")
        
    def test_inheritance(self):
        self.failUnless(isinstance(Echo(),MethodFilter))

class TestMemoryLogger(TestCase):
    
    def setUp(self):
        from xlutils.filter import MemoryLogger
        self.filter = MemoryLogger('somepath',['workbook'])

    @replace('xlutils.filter.guppy',True)
    @replace('xlutils.filter.hpy',Mock(),strict=False)
    def test_method(self,hpy):
        # XXX what are we logging?
        self.filter.method('name','foo',1)
        # hpy().heap().stat.dump('somepath')
        compare(hpy.call_args_list,[((),{})])
        hpy_i = hpy.return_value
        compare(hpy_i.method_calls,[('heap',(),{})])
        h = hpy_i.heap.return_value
        compare(h.method_calls,[('stat.dump', ('somepath',),{})])
    
    @replace('xlutils.filter.guppy',False)
    @replace('xlutils.filter.hpy',Mock(),strict=False)
    def test_method_no_heapy(self,hpy):
        self.filter.method('name','foo',1)
        compare(hpy.call_args_list,[])
    
    def test_inheritance(self):
        self.failUnless(isinstance(self.filter,MethodFilter))

from xlutils.filter import ErrorFilter

class TestErrorFilter(TestCase):

    def test_open_workbook_args(self):
        r = TestReader(('Sheet1',[['X']]))
        f = ErrorFilter()
        m = Mock()
        process(r,f,m)
        compare(m.method_calls,[
            ('start',(),{}),
            ('workbook',(C('xlrd.Book',
                           formatting_info=1,
                           on_demand=False,
                           ragged_rows=True,
                           strict=False),'test.xls'),{}),
            ('sheet',(C('xlrd.sheet.Sheet'),u'Sheet1'),{}),
            ('row',(0,0),{}),
            ('cell',(0,0,0,0),{}),
            ('finish',(),{}),
            ])
        
    @log_capture()
    def test_set_rdsheet_1(self,h):
        r = TestReader(
            ('Sheet1',[['S1R0C0']]),
            ('Sheet2',[[(XL_CELL_ERROR,0)]]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ErrorFilter()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.cell(0,0,0,0)
        f.set_rdsheet(book.sheet_by_index(1))
        f.cell(0,0,1,0)
        f.finish()
        compare(c.method_calls,[])
        h.check(
            ('xlutils.filter',
             'ERROR',
             va("Cell A1 of sheet b'Sheet2' contains a bad value: error (#NULL!)")),
            ('xlutils.filter',
             'ERROR',
             'No output as errors have occurred.'),
            )

    @log_capture()
    def test_set_rdsheet_2(self,h):
        r = TestReader(
            ('Sheet1',[['S1R0C0']]),
            ('Sheet2',[[(XL_CELL_ERROR,0)]]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ErrorFilter()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.cell(0,0,0,0)
        f.cell(0,0,1,0)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlrd.Book'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='new',strict=False), u'new'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('row', (1, 1),{}),
            ('cell', (1, 0, 1, 0),{}),
            ('finish', (), {})
            ])
        self.assertEqual(len(h.records),0)
    
    @log_capture()
    def test_multiple_workbooks_with_same_name(self,h):
        r = TestReader(
            ('Sheet1',[['S1R0C0']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ErrorFilter()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new1')
        f.cell(0,0,0,0)
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new2')
        f.cell(0,0,0,0)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlrd.Book'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='new1',strict=False), u'new1'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('workbook', (C('xlrd.Book'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='new2',strict=False), u'new2'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('finish', (), {})
            ])
        self.assertEqual(len(h.records),0)
    
    def test_finish_resets(self):
        # ...that's `start`s job!
        r = TestReader(
            ('Sheet1',[[(XL_CELL_ERROR,0)]]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ErrorFilter()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new1')
        f.cell(0,0,0,0)
        self.assertTrue(f.handler.fired)
        f.finish()
        compare(c.method_calls,[])
        self.assertFalse(f.handler.fired)
        compare(f.temp_path,None)

    @tempdir()
    def test_start(self,d):
        f = ErrorFilter()
        f.next = m = Mock()
        f.wtbook = 'junk'
        f.handler.fired = 'junk'
        f.temp_path = d.path
        f.prefix = 'junk'
        j = open(os.path.join(d.path,'junk.xls'),'wb')
        j.write(b'junk')
        j.close()

        f.start()

        compare(f.wtbook,None)
        compare(f.handler.fired,False)
        self.failIf(os.path.exists(d.path))
        compare(os.listdir(f.temp_path),[])
        compare(f.prefix,0)

        f.finish()
        
        compare(m.method_calls,[
            ('start', (), {}),
            ('finish', (), {})
            ])

    @log_capture()
    def test_no_error_on_bools(self,h):
        r = TestReader(
            ('Sheet',[[(XL_CELL_BOOLEAN,True)]]),
            )
        # fire methods on filter
        f = ErrorFilter()
        c = Mock()
        process(r,f,c)
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlrd.Book'), 'test.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='Sheet',strict=False), u'Sheet'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('finish', (), {})
            ])
        self.assertEqual(len(h.records),0)
    
from xlutils.filter import ColumnTrimmer

class TestColumnTrimmer(TestCase):

    @log_capture()
    def test_set_rdsheet_cols(self,h):
        r = TestReader(
            ('Sheet1',[['X',' ']]),
            ('Sheet2',[['X','X']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ColumnTrimmer()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.row(0,0)
        f.cell(0,0,0,0)
        f.set_rdsheet(book.sheet_by_index(1))
        f.cell(0,0,0,1)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='Sheet1',strict=False), u'new'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('set_rdsheet', (C('xlrd.sheet.Sheet',name='Sheet2',strict=False),),{}),
            ('cell', (0, 0, 0, 1),{}),
            ('finish', (), {})
            ])
        self.assertEqual(len(h.records),0)

    def test_set_rdsheet_rows(self):
        r = TestReader(
            ('Sheet1',[['X',' ']]),
            ('Sheet2',[['X','X'],['X','X'],['X','X']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ColumnTrimmer()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.row(0,0)
        f.cell(0,0,0,0)
        f.set_rdsheet(book.sheet_by_index(1))
        f.cell(2,0,1,0)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='Sheet1',strict=False), u'new'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('set_rdsheet', (C('xlrd.sheet.Sheet',name='Sheet2',strict=False),),{}),
            ('cell', (2, 0, 1, 0),{}),
            ('finish', (), {})
            ])

    def test_set_rdsheet_trim(self):
        r = TestReader(
            ('Sheet1',[['X',' ']]),
            ('Sheet2',[['X','X']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ColumnTrimmer()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.row(0,0)
        f.cell(0,0,0,0)
        f.cell(0,1,0,1)
        f.set_rdsheet(book.sheet_by_index(1))
        f.cell(0,0,1,0)
        f.cell(0,1,1,1)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='Sheet1',strict=False), u'new'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('cell', (0, 1, 0, 1),{}),
            ('set_rdsheet', (C('xlrd.sheet.Sheet',name='Sheet2',strict=False),),{}),
            ('cell', (0, 0, 1, 0),{}),
            ('cell', (0, 1, 1, 1),{}),
            ('finish', (), {})
            ])

    @log_capture()
    def test_use_write_sheet_name_in_logging(self,h):
        r = TestReader(
            ('Sheet1',[['X',' ']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on filter
        f = ColumnTrimmer()
        f.next = c = Mock()
        f.start()
        f.workbook(book,'new.xls')
        f.sheet(book.sheet_by_index(0),'new')
        f.row(0,0)
        f.cell(0,0,0,0)
        f.cell(0,1,0,1)
        f.finish()
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'new.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet',name='Sheet1',strict=False), u'new'),{}),
            ('row', (0, 0),{}),
            ('cell', (0, 0, 0, 0),{}),
            ('finish', (),{})
            ])
        h.check((
            'xlutils.filter',
            'DEBUG',
            va("Number of columns trimmed from 2 to 1 for sheet b'new'")
        ))

    @log_capture()
    def test_multiple_books(self,h):
        r = GlobReader(os.path.join(test_files,'test*.xls'))
        # fire methods on filter
        f = ColumnTrimmer()
        f.next = c = Mock()
        r(f)
        compare(c.method_calls,[
            ('start', (), {}),
            ('workbook', (C('xlrd.Book'), 'test.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet1'),{}),
            ('row', (0, 0),{}),
            ('row', (1, 1),{}),
            ('cell', (0, 0, 0, 0),{}),('cell', (0, 1, 0, 1),{}),
            ('cell', (1, 0, 1, 0),{}),('cell', (1, 1, 1, 1),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet2'),{}),
            ('row', (0, 0),{}),
            ('row', (1, 1),{}),
            ('cell', (0, 0, 0, 0),{}),('cell', (0, 1, 0, 1),{}),
            ('cell', (1, 0, 1, 0),{}),('cell', (1, 1, 1, 1),{}),
            ('workbook', (C('xlrd.Book'), 'testall.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet1'),{}),
            ('row', (0, 0),{}),
            ('row', (1, 1),{}),
            ('row', (2, 2),{}),
            ('row', (3, 3),{}),
            ('row', (4, 4),{}),
            ('row', (5, 5),{}),
            ('cell', (0, 0, 0, 0),{}),('cell', (0, 1, 0, 1),{}),
            ('cell', (1, 0, 1, 0),{}),('cell', (1, 1, 1, 1),{}),
            ('cell', (2, 0, 2, 0),{}),
            ('cell', (3, 0, 3, 0),{}),
            ('cell', (5, 0, 5, 0),{}),('cell', (5, 1, 5, 1),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet2'),{}),
            ('row', (0, 0),{}),
            ('row', (1, 1),{}),
            ('cell', (0, 0, 0, 0),{}),('cell', (0, 1, 0, 1),{}),
            ('cell', (1, 0, 1, 0),{}),('cell', (1, 1, 1, 1),{}),
            ('workbook', (C('xlrd.Book'), 'testnoformatting.xls'), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet1'), {}),
            ('row', (0, 0), {}),
            ('row', (1, 1), {}),
            ('row', (2, 2), {}),
            ('row', (3, 3), {}),
            ('row', (4, 4), {}),
            ('row', (5, 5), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell', (0, 1, 0, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell', (1, 1, 1, 1), {}),
            ('cell', (2, 0, 2, 0), {}),
            ('cell', (5, 0, 5, 0), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), u'Sheet2'), {}),
            ('row', (0, 0), {}),
            ('row', (1, 1), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell', (0, 1, 0, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell', (1, 1, 1, 1), {}),
            ('finish', (), {})
            ])
        self.assertEqual(len(h.records),0)

    def test_start(self):
        f = ColumnTrimmer()
        f.next = m = Mock()
        f.rdsheet = 'junk'
        f.pending_rdsheet = 'junk'
        f.ranges = 'junk'
        f.max_nonjunk = 'junk'
        f.max = 'junk'

        f.start()

        compare(f.rdsheet,None)
        compare(f.pending_rdsheet,None)
        compare(f.ranges,[])
        compare(f.max_nonjunk,0)
        compare(f.max,0)

        compare(m.method_calls,[
            ('start', (), {})
            ])
    
class CloseableTemporaryFile:
    def __init__(self,parent,filename):
        self.file = TemporaryFile()
        self.parent=parent
        self.filename=filename
    def close(self):
        self.parent.closed.add(self.filename)
        self.file.seek(0)
    def write(self,*args,**kw):
        self.file.write(*args,**kw)
    def real_close(self):
        self.file.close()
        
class TestWriter(BaseWriter):

    def __init__(self):
        self.files = {}
        self.closed = set()
        
    def get_stream(self,filename):
        f = CloseableTemporaryFile(self,filename)
        self.files[filename]=f
        return f
        
class TestBaseWriter(TestCase):

    def note_index(self,ao,eo,name):
        if name not in self.noted_indexes:
            self.noted_indexes[name]={}
        mapping = self.noted_indexes[name]
        a,e = getattr(ao,name,None),getattr(eo,name,None)
        if a not in mapping:
            mapping[a]=set()
        # for style compression, we may get multiple expected indexes
        # for each actual index in the output file. We just need to make
        # sure all the data is the same.
        mapping.get(a).add(e)
        
    def check_file(self,writer,path,
                   l_a_xf_list=20,
                   l_e_xf_list=26,
                   l_a_format_map=76,
                   l_e_format_map=75,
                   l_a_font_list=9,
                   l_e_font_list=7,
                   **provided_overrides):
        # allow overrides to be specified, as well as
        # default overrides, to save typing
        overrides = {
            'sheet':dict(
                # BUG: xlwt does nothing with col_default_width, it should :-(
                defcolwidth=None,
                )
            }
        for k,d in provided_overrides.items():
            if k in overrides:
                overrides[k].update(d)
            else:
                overrides[k]=d
                
        self.noted_indexes = {}
        # now open the source file
        e = open_workbook(path, formatting_info=1)
        # and the target file
        f = writer.files[os.path.split(path)[1]].file
        a = open_workbook(file_contents=f.read(), formatting_info=1)
        f.close()
        # and then compare
        def assertEqual(e,a,overrides,t,*names):
            for name in names:
                ea = overrides.get(t,{}).get(name,getattr(e,name))
                aa = getattr(a,name)
                self.assertEqual(aa,ea,'%s: %r(actual)!=%r(expected)'%(name,aa,ea))

        assertEqual(e, a, overrides, 'book',
                    'nsheets',
                    'datemode')
        
        for sheet_x in range(a.nsheets):
            ash = a.sheet_by_index(sheet_x)
            es = e.sheet_by_index(sheet_x)
            
            # order doesn't matter in this list
            compare(sorted(ash.merged_cells),sorted(es.merged_cells))

            assertEqual(
                es,ash,overrides,'sheet',
                'show_formulas',
                'show_grid_lines',
                'show_sheet_headers',
                'panes_are_frozen',
                'show_zero_values',
                'automatic_grid_line_colour',
                'columns_from_right_to_left',
                'show_outline_symbols',
                'remove_splits_if_pane_freeze_is_removed',
                'sheet_selected',
                'sheet_visible',
                'show_in_page_break_preview',
                'first_visible_rowx',
                'first_visible_colx',
                'gridline_colour_index',
                'cooked_page_break_preview_mag_factor',
                'cooked_normal_view_mag_factor',
                'default_row_height',
                'default_row_height_mismatch',
                'default_row_hidden',
                'default_additional_space_above',
                'default_additional_space_below',
                'nrows',
                'ncols',
                'standardwidth',
                'vert_split_pos',
                'horz_split_pos',
                'vert_split_first_visible',
                'horz_split_first_visible',
                'split_active_pane',
                )
            for col_x in range(ash.ncols):
                ac = ash.colinfo_map.get(col_x)
                ec = es.colinfo_map.get(col_x)
                if ac is not None:
                    assertEqual(ec,ac,overrides,'col',
                                'width',
                                'hidden',
                                'outline_level',
                                'collapsed',
                                )
                self.note_index(ac,ec,'xf_index')
            for row_x in range(ash.nrows):
                ar = ash.rowinfo_map.get(row_x)
                er = es.rowinfo_map.get(row_x)
                if er is None:
                    # NB: wlxt always writes Rowinfos, even
                    #     if none is supplied.
                    #     So, they end up with default values
                    #     which is what this tests
                    er = ar.__class__
                else:
                    assertEqual(er,ar,overrides,'row',
                                'height',
                                'has_default_height',
                                'height_mismatch',
                                'outline_level',
                                'outline_group_starts_ends',
                                'hidden',
                                'additional_space_above',
                                'additional_space_below',
                                'has_default_xf_index',
                                )
                    if ar.has_default_xf_index:
                        self.note_index(ar,er,'xf_index')
                for col_x in range(ash.ncols):
                    ac = ash.cell(row_x,col_x)
                    ec = es.cell(row_x,col_x)
                    assertEqual(ec,ac,overrides,'cell',
                                'ctype',
                                'value')
                    self.note_index(ac,ec,'xf_index')

        # only XFs that are in use are copied,
        # but we check those copied are identical
        self.assertEqual(len(a.xf_list),l_a_xf_list)
        self.assertEqual(len(e.xf_list),l_e_xf_list)
        for ai,eis in self.noted_indexes['xf_index'].items():
            if ai is None:
                continue
            axf = a.xf_list[ai]
            for ei in eis:
                exf = e.xf_list[ei]
                self.note_index(axf,exf,'format_key')
                self.note_index(axf,exf,'font_index')
                ap = axf.protection
                ep = exf.protection
                assertEqual(ep,ap,overrides,'protection',
                            'cell_locked',
                            'formula_hidden',
                            )
                ab = axf.border
                eb = exf.border
                assertEqual(eb,ab,overrides,'border',
                            'left_line_style',
                            'right_line_style',
                            'top_line_style',
                            'bottom_line_style',
                            'diag_line_style',
                            'left_colour_index',
                            'right_colour_index',
                            'top_colour_index',
                            'bottom_colour_index',
                            'diag_colour_index',
                            'diag_down',
                            'diag_up',
                            )
                ab = axf.background
                eb = exf.background
                assertEqual(eb,ab,overrides,'background',
                            'fill_pattern',
                            'pattern_colour_index',
                            'background_colour_index',
                            )
                aa = axf.alignment
                ea = exf.alignment
                assertEqual(ea,aa,overrides,'alignment',
                            'hor_align',
                            'vert_align',
                            'text_direction',
                            'rotation',
                            'text_wrapped',
                            'shrink_to_fit',
                            'indent_level',
                            )
            
        # xlwt writes more formats than exist in an original,
        # but we check those copied are identical
        self.assertEqual(len(a.format_map), l_a_format_map)
        self.assertEqual(len(e.format_map), l_e_format_map)
        for ai,eis in self.noted_indexes['format_key'].items():
            af = a.format_map[ai]
            for ei in eis:
                ef = e.format_map[ei]
                assertEqual(ef,af,overrides,'format',
                            'format_str',
                            'type')
        # xlwt writes more fonts than exist in an original,
        # but we check those that exist in both...
        self.assertEqual(len(a.font_list),l_a_font_list)
        self.assertEqual(len(e.font_list),l_e_font_list)
        for ai,eis in self.noted_indexes['font_index'].items():
            af = a.font_list[ai]
            for ei in eis:
                ef = e.font_list[ei]
                assertEqual(ef,af,overrides,'font',
                            'height',
                            'italic',
                            'struck_out',
                            'outline',
                            'colour_index',
                            'bold',
                            'weight',
                            'escapement',
                            'underline_type',
                            'family',
                            'character_set',
                            'name',
                            )

    def test_single_workbook_with_all_features(self):
        # create test reader
        test_xls_path = os.path.join(test_files,'testall.xls')
        r = GlobReader(test_xls_path)
        # source sheet must have merged cells for test!
        book = tuple(r.get_workbooks())[0][0]
        self.failUnless(book.sheet_by_index(0).merged_cells)
        # source book must also have a sheet other than the
        # first one selected
        compare([s.sheet_selected for s in book.sheets()],[0,1])
        compare([s.sheet_visible for s in book.sheets()],[0,1])
        # source book must have show zeros set appropriately:
        compare([s.show_zero_values for s in book.sheets()],[0,0])
        # send straight to writer
        w = TestWriter()
        r(w)
        # check stuff on the writer
        compare(w.files.keys(), expected=['testall.xls'])
        self.failUnless('testall.xls' in w.closed)
        self.check_file(w,test_xls_path)

    def test_dates(self):
        # create test reader
        xls_path = os.path.join(test_files, 'date.xls')
        r = GlobReader(xls_path)
        # source book must be in weird date mode
        book = tuple(r.get_workbooks())[0][0]
        self.assertEqual(book.datemode, 1)
        # send straight to writer
        w = TestWriter()
        r(w)
        self.check_file(w, xls_path,
                        # date.xls is fewer sheets and styles than the default
                        l_a_xf_list=19,
                        l_e_xf_list=65,
                        l_e_format_map=74,
                        l_a_font_list=7,
                        l_e_font_list=25,
                        # dates.xls had a standardwidth record but xlwt can't
                        # currently write these :-/
                        sheet=dict(standardwidth=None),)

    def test_single_workbook_no_formatting(self):
        # create test reader
        test_xls_path = os.path.join(test_files,'testnoformatting.xls')
        r = XLRDReader(open_workbook(os.path.join(test_files,'testall.xls')),'testnoformatting.xls')
        # source sheet must have merged cells for test!
        # send straight to writer
        w = TestWriter()
        r(w)
        # check stuff on the writer
        compare(w.files.keys(), expected=['testnoformatting.xls'])
        self.failUnless('testnoformatting.xls' in w.closed)
        self.check_file(w,test_xls_path,
                        l_a_xf_list=17,
                        l_e_xf_list=17,
                        l_a_format_map=75,
                        l_a_font_list=6,
                        l_e_font_list=6)

    def test_multiple_workbooks(self):
        # globreader is tested elsewhere
        r = GlobReader(os.path.join(test_files,'test*.xls'))
        # send straight to writer
        w = TestWriter()
        r(w)
        # check stuff on the writer
        compare(
            sorted(w.files.keys()),
            expected=['test.xls', 'testall.xls', 'testnoformatting.xls']
        )
        self.failUnless('test.xls' in w.closed)
        self.failUnless('testall.xls' in w.closed)
        self.failUnless('testnoformatting.xls' in w.closed)
        self.check_file(w,os.path.join(test_files,'testall.xls'))
        self.check_file(w,os.path.join(test_files,'test.xls'),
                        18,21,76,75,7,4)
        self.check_file(w,os.path.join(test_files,'testnoformatting.xls'),
                        18,17,75,75,6,6)
    
    def test_start(self):
        w = TestWriter()
        w.wtbook = 'junk'
        w.start()
        compare(w.wtbook,None)
    
    @replace('xlutils.filter.BaseWriter.close',Mock())
    def test_workbook(self,c):
        # style copying is tested in the more complete tests
        # here we just check that certain atributes are set properly
        w = TestWriter()
        w.style_list = 'junk'
        w.wtsheet_names = 'junk'
        w.wtsheet_index = 'junk'
        w.sheet_visible = 'junk'
        b = make_book()
        w.workbook(b,'foo')
        compare(c.call_args_list,[((),{})])
        compare(w.rdbook,b)
        compare(w.wtbook,C('xlwt.Workbook'))
        compare(w.wtname,'foo')
        compare(w.style_list,[])
        compare(w.wtsheet_names,set())
        compare(w.wtsheet_index,0)
        compare(w.sheet_visible,False)
    
    def test_set_rd_sheet(self):
        # also tests that 'row' doesn't have to be called,
        # only cell
        r = TestReader(
            ('Sheet1',(('S1R0C0',),
                       ('S1R1C0',),)),
            ('Sheet2',(('S2R0C0',),
                       ('S2R1C0',),)),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on writer
        w = TestWriter()
        w.start()
        w.workbook(book,'new.xls')
        w.sheet(book.sheet_by_index(0),'new')
        w.row(0,0)
        w.cell(0,0,0,0)
        w.set_rdsheet(book.sheet_by_index(1))
        w.cell(0,0,1,0)
        w.set_rdsheet(book.sheet_by_index(0))
        w.cell(1,0,2,0)
        w.set_rdsheet(book.sheet_by_index(1))
        w.cell(1,0,3,0)
        w.finish()
        # check everything got written and closed
        compare(w.files.keys(), expected=['new.xls'])
        self.failUnless('new.xls' in w.closed)
        # now check the cells written
        f = w.files['new.xls'].file
        a = open_workbook(file_contents=f.read(), formatting_info=1)
        self.assertEqual(a.nsheets,1)
        sheet = a.sheet_by_index(0)
        self.assertEqual(sheet.nrows,4)
        self.assertEqual(sheet.ncols,1)
        self.assertEqual(sheet.cell(0,0).value,'S1R0C0')
        self.assertEqual(sheet.cell(1,0).value,'S2R0C0')
        self.assertEqual(sheet.cell(2,0).value,'S1R1C0')
        self.assertEqual(sheet.cell(3,0).value,'S2R1C0')
        
    def test_bogus_sheet_name(self):
        r = TestReader(
            ('sheet',([['S1R0C0']]),),
            ('Sheet',([['S2R0C0']]),),
            )
        # fire methods on writer
        with ShouldRaise(ValueError(
            "A sheet named 'sheet' has already been added!"
            )):
            r(TestWriter())
    
    def test_empty_sheet_name(self):
        r = TestReader(
            ('',([['S1R0C0']]),),
            )
        # fire methods on writer
        with ShouldRaise(ValueError(
            'Empty sheet name will result in invalid Excel file!'
            )):
            r(TestWriter())
    
    def test_max_length_sheet_name(self):
        name = 'X'*31
        r = TestReader(
            (name,([['S1R0C0']]),),
            )
        w = TestWriter()
        r(w)
        compare(w.files.keys(), expected=['test.xls'])
        f = w.files['test.xls'].file
        a = open_workbook(file_contents=f.read(), formatting_info=1)
        self.assertEqual(a.sheet_names(),[name])
        
    def test_panes(self):
        r = TestReader()
        r.formatting_info = True
        
        r.setup(('sheet',[['S1R0C0']]))
        
        book = tuple(r.get_workbooks())[0][0]
        sheet = book.sheet_by_index(0)
        sheet.panes_are_frozen = 1
        sheet.has_pane_record = True
        sheet.vert_split_pos = 1
        sheet.horz_split_pos = 2
        sheet.vert_split_first_visible = 3
        sheet.horz_split_first_visible = 4
        sheet.split_active_pane = 3
        
        w = TestWriter()
        r(w)
        
        compare(w.files.keys(), expected=['test.xls'])
        f = w.files['test.xls'].file
        a = open_workbook(file_contents=f.read(),formatting_info=1)
        sheet = a.sheet_by_index(0)
        self.assertEqual(1,sheet.panes_are_frozen)
        self.assertEqual(1,sheet.has_pane_record)
        self.assertEqual(1,sheet.vert_split_pos)
        self.assertEqual(2,sheet.horz_split_pos)
        self.assertEqual(3,sheet.vert_split_first_visible)
        self.assertEqual(4,sheet.horz_split_first_visible)
        # for splits, this is a magic value, and is computed
        # by xlwt.
        self.assertEqual(0,sheet.split_active_pane)
        
    def test_splits(self):
        r = TestReader()
        r.formatting_info = True
        
        r.setup(('sheet',[['S1R0C0']]))
        
        book = tuple(r.get_workbooks())[0][0]
        sheet = book.sheet_by_index(0)
        sheet.panes_are_frozen = 0
        sheet.has_pane_record = True
        sheet.vert_split_pos = 1
        sheet.horz_split_pos = 2
        sheet.vert_split_first_visible = 3
        sheet.horz_split_first_visible = 4
        sheet.split_active_pane = 3
        
        w = TestWriter()
        r(w)
        
        compare(w.files.keys(), expected=['test.xls'])
        f = w.files['test.xls'].file
        a = open_workbook(file_contents=f.read(),formatting_info=1)
        sheet = a.sheet_by_index(0)
        self.assertEqual(0,sheet.panes_are_frozen)
        self.assertEqual(1,sheet.has_pane_record)
        self.assertEqual(1,sheet.vert_split_pos)
        self.assertEqual(2,sheet.horz_split_pos)
        self.assertEqual(3,sheet.vert_split_first_visible)
        self.assertEqual(4,sheet.horz_split_first_visible)
        self.assertEqual(3,sheet.split_active_pane)
        
    def test_zoom_factors(self):
        r = TestReader()
        r.formatting_info = True
        
        r.setup(('sheet',[['S1R0C0']]))
        
        book = tuple(r.get_workbooks())[0][0]
        sheet = book.sheet_by_index(0)
        sheet.cooked_normal_view_mag_factor = 33
        sheet.cooked_page_break_preview_mag_factor = 44
        sheet.show_in_page_break_preview = True

        w = TestWriter()
        r(w)
        
        compare(w.files.keys(), expected=['test.xls'])
        f = w.files['test.xls'].file
        a = open_workbook(file_contents=f.read(),formatting_info=1)
        sheet = a.sheet_by_index(0)
        self.assertEqual(33,sheet.cooked_normal_view_mag_factor)
        self.assertEqual(44,sheet.cooked_page_break_preview_mag_factor)
        self.assertEqual(1,sheet.show_in_page_break_preview)
        
    def test_excessive_length_sheet_name(self):
        r = TestReader(
            ('X'*32,([['S1R0C0']]),),
            )
        # fire methods on writer
        with ShouldRaise(ValueError(
            'Sheet name cannot be more than 31 characters long, '
            'supplied name was 32 characters long!'
            )):
            r(TestWriter())

    def test_copy_error_cells(self):
        r = TestReader(
            ('Errors',([[(XL_CELL_ERROR,0)]]),),
            )
        w = TestWriter()
        r(w)
        compare(w.files.keys(), expected=['test.xls'])
        a = open_workbook(file_contents=w.files['test.xls'].file.read())
        cell = a.sheet_by_index(0).cell(0,0)
        self.assertEqual(cell.ctype,XL_CELL_ERROR)
        self.assertEqual(cell.value,0)
    
    def test_copy_boolean_cells(self):
        r = TestReader(
            ('Bools',([[(XL_CELL_BOOLEAN,True)]]),),
            )
        w = TestWriter()
        r(w)
        compare(w.files.keys(), expected=['test.xls'])
        a = open_workbook(file_contents=w.files['test.xls'].file.read())
        cell = a.sheet_by_index(0).cell(0,0)
        self.assertEqual(cell.ctype,XL_CELL_BOOLEAN)
        self.assertEqual(cell.value,True)
    
class TestDirectoryWriter(TestCase):

    def test_inheritance(self):
        from xlutils.filter import DirectoryWriter
        self.failUnless(isinstance(DirectoryWriter('foo'),BaseWriter))

    @tempdir()
    def test_plus_in_workbook_name(self,d):
        from xlutils.filter import DirectoryWriter
        r = TestReader(
            ('Sheet1',[['Cell']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        # fire methods on writer
        w = DirectoryWriter(d.path)
        w.start()
        w.workbook(book,'a+file.xls')
        w.sheet(book.sheet_by_index(0),'new')
        w.row(0,0)
        w.cell(0,0,0,0)
        w.finish()
        # check file exists with the right name
        self.assertEqual(os.listdir(d.path),['a+file.xls'])

class TestXLWTWriter(TestCase):

    def setUp(self):
        self.w = XLWTWriter()
        
    def test_inheritance(self):
        self.failUnless(isinstance(self.w,BaseWriter))

    def test_no_files(self):
        r = GlobReader(os.path.join(test_files,'*not.xls'))
        r(self.w)
        compare(self.w.output,[])
        
    def test_one_file(self):
        r = GlobReader(os.path.join(test_files,'test.xls'))
        r(self.w)
        compare(self.w.output,[
            ('test.xls',C('xlwt.Workbook'))
            ])
        # make sure wtbook is deleted
        compare(self.w.wtbook,None)
        
    def test_multiple_files(self):
        r = GlobReader(os.path.join(test_files,'test*.xls'))
        r(self.w)
        compare(self.w.output,[
            ('test.xls',C('xlwt.Workbook')),
            ('testall.xls',C('xlwt.Workbook')),
            ('testnoformatting.xls',C('xlwt.Workbook')),
            ])
        
    def test_multiple_files_same_name(self):
        r = TestReader(
            ('Sheet1',[['S1R0C0']]),
            )
        book = tuple(r.get_workbooks())[0][0]
        self.w.start()
        self.w.workbook(book,'new.xls')
        self.w.sheet(book.sheet_by_index(0),'new1')
        self.w.cell(0,0,0,0)
        self.w.workbook(book,'new.xls')
        self.w.sheet(book.sheet_by_index(0),'new2')
        self.w.cell(0,0,0,0)
        self.w.finish()
        compare(self.w.output,[
            ('new.xls',C('xlwt.Workbook')),
            ('new.xls',C('xlwt.Workbook')),
            ])
        compare(self.w.output[0][1].get_sheet(0).name,
                'new1')
        compare(self.w.output[1][1].get_sheet(0).name,
                'new2')

class TestProcess(TestCase):

    def test_setup(self):
        class DummyReader:
            def __call__(self,filter):
                filter.finished()
        F1 = Mock()
        F2 = Mock()
        process(DummyReader(),F1,F2)
        self.failUnless(F1.next is F2)
        self.failUnless(isinstance(F2.next,Mock))
        compare(F1.method_calls,[('finished',(),{})])
        compare(F2.method_calls,[])
    
class TestTestReader(TestCase):

    def test_cell_type(self):
        r = TestReader(('Sheet1',(((XL_CELL_NUMBER,0.0),),)))
        book = tuple(r.get_workbooks())[0][0]
        cell = book.sheet_by_index(0).cell(0,0)
        self.assertEqual(cell.value,0.0)
        self.assertEqual(cell.ctype,XL_CELL_NUMBER)
        
    def test(self):
        r = TestReader(
            test1=[('Sheet1',[['R1C1','R1C2'],
                              ['R2C1','R2C2']]),
                   ('Sheet2',[['R3C1','R3C2'],
                              ['R4C1','R4C2']])],
            test2=[('Sheet3',[['R5C1','R5C2'],
                              ['R6C1','R6C2']]),
                   ('Sheet4',[['R7C1','R7C2'],
                              ['R8C1','R8C2']])],
            )
        f = Mock()
        r(f)
        compare([
            ('start', (), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'test1.xls'), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet1'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell',(0, 1, 0, 1), {}),
            ('row', (1, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell',(1, 1, 1, 1), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet2'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell',(0, 1, 0, 1), {}),
            ('row', (1, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell',(1, 1, 1, 1), {}),
            ('workbook', (C('xlutils.tests.fixtures.DummyBook'), 'test2.xls'), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet3'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell',(0, 1, 0, 1), {}),
            ('row', (1, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell',(1, 1, 1, 1), {}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet4'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell',(0, 1, 0, 1), {}),
            ('row', (1, 1), {}),
            ('cell', (1, 0, 1, 0), {}),
            ('cell',(1, 1, 1, 1), {}),
            ('finish', (), {})
            ],f.method_calls)
        
    def test_setup(self):
        r = TestReader()
        f = Mock()
        r(f)
        compare([('start', (), {}), ('finish', (), {})],f.method_calls)
        r.setup(('Sheet1',[['R1C1']]),
                test1=[('Sheet2',[['R2C1']])])
        r(f)
        compare([
            ('start', (), {}),
            ('finish', (), {}),
            ('start', (), {}),
            ('workbook',(C('xlutils.tests.fixtures.DummyBook'), 'test.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet1'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('workbook',(C('xlutils.tests.fixtures.DummyBook'), 'test1.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet2'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('finish', (), {})],f.method_calls)
        
    def test_formatting_info(self):
        r = TestReader()
        f = Mock()
        
        r.formatting_info = True
        
        r.setup(('Sheet1',[['R1C1','R1C2']]))

        # at this point you can now manipulate the xf index as follows:
        book = r.books[0][0]
        sx,rx,cx = 0,0,0
        book.sheet_by_index(sx)._cell_xf_indexes[rx][cx]=42
        # NB: cells where you haven't specified an xf index manually as
        #     above will have an xf index of 0:
        compare(book.sheet_by_index(0).cell(0,1).xf_index,0)

        # NB: when formattng is turned on, an XF will be created at index zero:
        compare(C(XF),book.xf_list[0])
        # ...but no others:
        with ShouldRaise(IndexError):
            book.xf_list[42]
        # so you'll need to manually create them if you need them.
        # See fixtures.py for examples.

        r(f)
        
        compare([
            ('start', (), {}),
            ('workbook',(C('xlutils.tests.fixtures.DummyBook'), 'test.xls'),{}),
            ('sheet', (C('xlrd.sheet.Sheet'), 'Sheet1'), {}),
            ('row', (0, 0), {}),
            ('cell', (0, 0, 0, 0), {}),
            ('cell', (0, 1, 0, 1), {}),
            ('finish', (), {})],f.method_calls)

        compare(book.sheet_by_index(0).cell(0,0).xf_index,42)
        
        
def test_suite():
    return TestSuite((
        makeSuite(TestBaseReader),
        makeSuite(TestTestReader),
        makeSuite(TestBaseFilter),
        makeSuite(TestMethodFilter),
        makeSuite(TestEcho),
        makeSuite(TestMemoryLogger),
        makeSuite(TestErrorFilter),
        makeSuite(TestColumnTrimmer),
        makeSuite(TestBaseWriter),
        makeSuite(TestDirectoryWriter),
        makeSuite(TestXLWTWriter),
        makeSuite(TestProcess),
        ))
