#!/usr/bin/tclsh
# vim: set ts=4 fdm=marker fmr=<<<,>>> cms=;#%s noexpandtab:
# the next line restarts using wish \
exec wish "$0" ${1+"$@"}

#######################################################################
#
# TkSQLite
#
# TkSQLite is a database manager for SQLite implemented by Tcl/Tk.
# It can manage SQLite2 and SQLite3 database. And it can treat many 
# encodings like Shift-JIS,  EUC-JP and Windows native milti-byte 
# encodings not only UTF-8. And it can run on Windows and Linux 
# and MacOSX as native Look & Feel. And it's Freeware. 
# 
#######################################################################
# ;#<<<
#
# Requirement
# ---------------------------------------------------------------------
# Tcl/Tk 8.4+
# tclsqlite  or|and  tclsqlite3
# Tktable
# Tile 0.7+
# Treectrl 2.1+
# Dict 8.5.1+
# 
# optionnal packages:
# Trf
# Tcllib
# tkImg
# 
# License
# ---------------------------------------------------------------------
set COPYRIGHT {
Copyright (c) 2004, 2005, 2006, 2007, 2008, 2009 OHTSUKA, Yoshio
This program is free to use, modify, extend at will. The author(s)
provides no warrantees, guarantees or any responsibility for usage.
Redistributions in any form must retain this copyright notice.
ohtsuka.yoshio@gmail.com
}
# 
# How to use
# ---------------------------------------------------------------------
# see
# http://reddog.s35.xrea.com/wiki/TkSQLite.html
# 
# ChangeLog
# ---------------------------------------------------------------------
# 2004-01-14 First Release.
# 2004-03-17 Sort table tree.
# 2004-08-12 Add encoding menu.
# 2005-05-13 Support sqlite3 and not sqlite2
# 2005-11-04 Version 0.2
#   * Support both SQLite2 and SQLite3.
#   * Support reading/writing data as any encoding that Tcl can use.
#   * Support attach database and temp table.
#   * Use Tk Theme Engine instead of BWidget.
#   * Improve TkTable behabior.
#   * Highlighting SQL command syntax.
#   * Release English version.
#   * Fix many bugs & Refactored.
# 2005-12-04 Version 0.3
#   * Create Table Dialog, Create Index Dialog and Preference Dialog.
#   * Improve Look & Feel.
#   * Sort Grid Table command.
#   * Search Dialog for Grid Table.
#   * Add Option of Vi like keybind for Grid Table.
#   * Attack/Detach db menu.
#   * Import/Export SQL and CSV(or Tab)
#   * Add many db functions(regexp, regsub, md5 etc).
# 2006-08-04 Version 0.4
#   * Require dict, Trf, tablelist and tkImg packages.
#   * Improve Keyword Highlight.
#   * Improve Preference, CreateTable and CreateIndex Dialogs.
#   * Add 3.x compatible format option.
#   * Add Cache size Dialog.
#   * Add DataEditor Dialog.
#   * Enable to change table schema.
#   * Enable to view images in column.(the format is depended on tkImg)
#   * Speed up base64 encode/decode.
#   * New user function compress, decompress, read_file, write_file
#   * Add binary mode on crypt user function (md5, base64_encode, etc)
# 2007-07-04 Version 0.5
#   * Require tktreectrl2.1+.
#   * Not Require tablelist.
#   * Add CreateTrigger Dialog.
#   * New message dialog (avoided to use ttk::dialog).
#   * Add new file selection dialog (http://wiki.tcl.tk/15897).
#   * Improved widgets L&F on WindowsXP at tile0.8.
#   * Enable to load loadable extension and user tcl script.
#   * Enable to create UTF-16le, UTF-16be database.
#
# Known Probrem
# ---------------------------------------------------------------------
# - If alter table command was used to view, we would lost the view.
#   But don't worry. The view was not dropped. Please re-open or use 
#   vacuum command. Then we would be able to find the view again. 
#   This is sqlite probrem. 
# 
# - If there were same named tables in attached or temp database, 
#   a column infomation of the table was not got correctly. This 
#   probrem would be occured on attached/temp table on sqlite2 and 
#   temp table on sqlite3.
#   We get column infomation from 'pragma table_info' command, 
#   but the command of sqlite2 doesn't have a way to distinguish 
#   between a table and an attached same named another table.
#   TODO:we must parse create statement ourself..?
#   See also:
#   http://www.sqlite.org/cvstrac/tktview?tn=1308
#
# - If database had a trigger, grid table edit does not update the updated 
#   rows by the trigger.  Grid Table editor update only one row. So if 
#   the trigger updated some rows in the editing table, we must update 
#   the table by hand.(Press F5 or Press u in vi mode)
#
# - On Linux all tk widgets disallow keyboard input, if encoding system 
#   is unicode.
#;#>>>
set VERSION 0.5.8
package require Tk 8.4
package require Tktable
if {[info tclversion] < 8.5} {
package require dict
}
package require treectrl 2.1
set tile_version [package require tile 0.7]
if {[join [lrange [split $tile_version .] 0 1] .] >= 0.8} {
	set tile_ns ttk
	# In tk8.5a6, style command is not imported yet.
	# But the cmd is already imported tile0.8(tk8.4).
	# so we need catch it.
	catch {namespace import ttk::style}
} else {
	set tile_ns tile
}
if {[tk windowingsystem] eq "x11"} {
	${tile_ns}::setTheme clam
}
# here is for debug
;#${tile_ns}::setTheme clam
;#${tile_ns}::setTheme default
set tile_currentTheme [set ${tile_ns}::currentTheme]

#
# Base64
#
if {[catch {package require Trf}]} {
# base64.tcl --
#
# Encode/Decode base64 for a string
# Stephen Uhler / Brent Welch (c) 1997 Sun Microsystems
# The decoder was done for exmh by Chris Garrigues
#
# Copyright (c) 1998-2000 by Ajuba Solutions.
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
# 
# RCS: @(#) $Id: base64.tcl 182 2006-07-24 09:20:41Z ir004961 $
#package provide base64 2.3.2

package require Tcl 8.4
namespace eval ::base64 {
    namespace export encode decode
}

namespace eval base64 {
variable base64 {}
variable base64_en {}

set i 0
foreach char {A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \
	a b c d e f g h i j k l m n o p q r s t u v w x y z \
	0 1 2 3 4 5 6 7 8 9 + /} {
	set base64_tmp($char) $i
	lappend base64_en $char
	incr i
}
scan z %c len
for {set i 0} {$i <= $len} {incr i} {
	set char [format %c $i]
	set val {}
	if {[info exists base64_tmp($char)]} {
	set val $base64_tmp($char)
	} else {
	set val {}
	}
	lappend base64 $val
}

scan = %c i
set base64 [lreplace $base64 $i $i -1]
unset base64_tmp i char len val

namespace export encode decode
}
proc ::base64::encode {args} {
set base64_en $::base64::base64_en
set wrapchar "\n"
set maxlen 60
if { [llength $args] == 0 } {
	error "wrong # args: should be \"[lindex [info level 0] 0]\
		?-maxlen maxlen? ?-wrapchar wrapchar? string\""
}
set optionStrings [list "-maxlen" "-wrapchar"]
for {set i 0} {$i < [llength $args] - 1} {incr i} {
	set arg [lindex $args $i]
	set index [lsearch -glob $optionStrings "${arg}*"]
	if { $index == -1 } {
	error "unknown option \"$arg\": must be -maxlen or -wrapchar"
	}
	incr i
	if { $i >= [llength $args] - 1 } {
	error "value for \"$arg\" missing"
	}
	set val [lindex $args $i]
	set [string range [lindex $optionStrings $index] 1 end] $val
}
if {[catch {expr {$maxlen % 2}}]} {
	error "expected integer but got \"$maxlen\""
}
set string [lindex $args end]
set result {}
set state 0
set length 0
binary scan $string c* X
foreach {x y z} $X {
	if {$maxlen && $length >= $maxlen} {
	append result $wrapchar
	set length 0
	}

	append result [lindex $base64_en [expr {($x >>2) & 0x3F}]] 
	if {$y != {}} {
	append result [lindex $base64_en [expr {(($x << 4) & 0x30) | (($y >> 4) & 0xF)}]] 
	if {$z != {}} {
		append result \
			[lindex $base64_en [expr {(($y << 2) & 0x3C) | (($z >> 6) & 0x3)}]]
		append result [lindex $base64_en [expr {($z & 0x3F)}]]
	} else {
		set state 2
		break
	}
	} else {
	set state 1
	break
	}
	incr length 4
}
if {$state == 1} {
	append result [lindex $base64_en [expr {(($x << 4) & 0x30)}]]== 
} elseif {$state == 2} {
	append result [lindex $base64_en [expr {(($y << 2) & 0x3C)}]]=  
}
return $result
}
proc ::base64::decode {string} {
if {[string length $string] == 0} {return ""}
set base64 $::base64::base64
set output "" ; # Fix for [Bug 821126]
binary scan $string c* X
foreach x $X {
	set bits [lindex $base64 $x]
	if {$bits >= 0} {
	if {[llength [lappend nums $bits]] == 4} {
		foreach {v w z y} $nums break
		set a [expr {($v << 2) | ($w >> 4)}]
		set b [expr {(($w & 0xF) << 4) | ($z >> 2)}]
		set c [expr {(($z & 0x3) << 6) | $y}]
		append output [binary format ccc $a $b $c]
		set nums {}
	}		
	} elseif {$bits == -1} {
	foreach {v w z} $nums break
	set a [expr {($v << 2) | (($w & 0x30) >> 4)}]
	if {$z == {}} {
		append output [binary format c $a ]
	} else {
		set b [expr {(($w & 0xF) << 4) | (($z & 0x3C) >> 2)}]
		append output [binary format cc $a $b]
	}		
	break
	} else {
	continue
	}
}
return $output
}
proc ::base64 {args} {
	set len [llength $args]
	for {set i 0} {$i < $len} {incr i} {
		switch -exact -- [lindex $args $i] {
			-mode {
				incr i
				switch -exact -- [lindex $args $i] {
					encode {set mode encode}
					decode {set mode decode}
				}
			}
			-- {
				incr i
				set data [lindex $args $i]
			}
		}
	}
	switch -exact -- $mode {
		encode { return [::base64::encode $data] }
		decode { return [::base64::decode $data] }
		default {error}
	}
}

}

#
# Locale Strings
#
package require msgcat
::msgcat::mcmset ja {
{OK} "OK"
{Cancel} "\u30ad\u30e3\u30f3\u30bb\u30eb"
{Yes} "\u306f\u3044"
{No} "\u3044\u3044\u3048"
{Retry} "\u518d\u8a66\u884c"
{Add} "\u8ffd\u52a0"
{Remove} "\u524a\u9664"
{Config} "\u8a2d\u5b9a"
{Configure} "\u8a2d\u5b9a"
{Table} "\u30c6\u30fc\u30d6\u30eb"
{View} "\u30d3\u30e5\u30fc"
{Index} "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9"
{Trigger} "\u30c8\u30ea\u30ac\u30fc"
{Event} "\u30a4\u30d9\u30f3\u30c8"
{Result} "\u5b9f\u884c\u7d50\u679c"
{Edit} "\u7de8\u96c6"
{Editable} "\u7de8\u96c6\u53ef"
{Not Editable} "\u7de8\u96c6\u4e0d\u53ef"
{File} "\u30d5\u30a1\u30a4\u30eb"
{Database} "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9"
{Tool} "\u30c4\u30fc\u30eb"
{Help} "\u30d8\u30eb\u30d7"
{New SQLite2 DB} "SQLite2DB\u3092\u65b0\u898f\u4f5c\u6210"
{New SQLite3 DB} "SQLite3DB\u3092\u65b0\u898f\u4f5c\u6210"
{Open} "\u958b\u304f"
{Recent Files} "\u6700\u8fd1\u958b\u3044\u305f\u30d5\u30a1\u30a4\u30eb"
{Change Current Dir} "\u73fe\u5728\u306e\u30c7\u30a3\u30ec\u30af\u30c8\u30ea\u306e\u5909\u66f4"
{Close} "\u9589\u3058\u308b"
{Attach} "\u30a2\u30bf\u30c3\u30c1"
{Detach} "\u30c7\u30bf\u30c3\u30c1"
{Import} "\u8aad\u307f\u8fbc\u307f"
{Export} "\u66f8\u304d\u51fa\u3057"
{SQL} "SQL"
{Text File} "\u30c6\u30ad\u30b9\u30c8\u30d5\u30a1\u30a4\u30eb"
{Exit} "\u7d42\u4e86"
{Run SQL} "SQL\u5b9f\u884c"
{Text} "\u30c6\u30ad\u30b9\u30c8"
{Image} "\u753b\u50cf"
{Save} "\u4fdd\u5b58"
{Load} "\u8aad\u307f\u8fbc\u307f"
{Show Create Table Statement} "\u30c6\u30fc\u30d6\u30eb\u4f5c\u6210\u306eSQL\u6587\u8868\u793a"
{Show Create Index Statement} "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9\u4f5c\u6210\u306eSQL\u6587\u8868\u793a"
{Show Create Trigger Statement} "\u30c8\u30ea\u30ac\u30fc\u4f5c\u6210\u306eSQL\u6587\u8868\u793a"
{Create Table} "\u30c6\u30fc\u30d6\u30eb\u4f5c\u6210"
{Crete View} "\u30d3\u30e5\u30fc\u4f5c\u6210"
{Create Index} "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9\u4f5c\u6210"
{Create Trigger} "\u30c8\u30ea\u30ac\u30fc\u4f5c\u6210"
{Drop Table/View} "\u30c6\u30fc\u30d6\u30eb/\u30d3\u30e5\u30fc\u524a\u9664"
{Drop Index} "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9\u524a\u9664"
{Drop Trigger} "\u30c8\u30ea\u30ac\u30fc\u524a\u9664"
{Modify Table Schema} "\u30c6\u30fc\u30d6\u30eb\u30b9\u30ad\u30fc\u30de\u306e\u5909\u66f4"
{Set Cache Size} "\u30ad\u30e3\u30c3\u30b7\u30e5\u30b5\u30a4\u30ba\u8a2d\u5b9a"
{VACUUM} "VACUUM"
{Check Integrity} "\u6574\u5408\u6027\u30c1\u30a7\u30c3\u30af"
{Encoding} "\u30a8\u30f3\u30b3\u30fc\u30c7\u30a3\u30f3\u30b0"
{Propery} "\u30d7\u30ed\u30d1\u30c6\u30a3"
{SQL Command} "SQL\u30b3\u30de\u30f3\u30c9"
{Command History} "\u30b3\u30de\u30f3\u30c9\u5c65\u6b74"
{Preference} "\u521d\u671f\u8a2d\u5b9a"
{Undo} "\u5143\u306b\u623b\u3059"
{Redo} "\u3084\u308a\u76f4\u3057"
{Cut} "\u5207\u308a\u53d6\u308a"
{Copy} "\u30b3\u30d4\u30fc"
{Paste} "\u8cbc\u308a\u4ed8\u3051"
{Delete} "\u524a\u9664"
{Select All} "\u5168\u3066\u9078\u629e"
{Delete All} "\u5168\u3066\u524a\u9664"
{Update} "\u66f4\u65b0"
{Sort Asc as Number} "\u6607\u9806\u3067\u30bd\u30fc\u30c8(\u6570\u5024)"
{Sort Desc as Number} "\u964d\u9806\u3067\u30bd\u30fc\u30c8(\u6570\u5024)"
{Sort Asc as String} "\u6607\u9806\u3067\u30bd\u30fc\u30c8(\u8f9e\u66f8\u9806)"
{Sort Desc as String} "\u964d\u9806\u3067\u30bd\u30fc\u30c8(\u8f9e\u66f8\u9806)"
{Reset Row Size} "\u884c\u306e\u9ad8\u3055\u3092\u30ea\u30bb\u30c3\u30c8"
{Reset Column Size} "\u5217\u306e\u5e45\u3092\u30ea\u30bb\u30c3\u30c8"
{Insert New Row} "\u65b0\u3057\u3044\u884c\u306e\u633f\u5165"
{Delete Row} "\u884c\u306e\u524a\u9664"
{Default Value} "\u30c7\u30d5\u30a9\u30eb\u30c8\u5024"
{Modify Column} "\u30ab\u30e9\u30e0\u5909\u66f4"
{Add Column} "\u30ab\u30e9\u30e0\u8ffd\u52a0"
{Delete Column} "\u30ab\u30e9\u30e0\u524a\u9664"
{Up} "\u4e0a"
{Down} "\u4e0b"
{Show Only} "\u8868\u793a\u306e\u307f"
{Name} "\u540d\u524d"
{Data Type} "\u30c7\u30fc\u30bf\u578b"
{Preview} "\u30d7\u30ec\u30d3\u30e5\u30fc"
{Table-Constraints} "\u30c6\u30fc\u30d6\u30eb\u5236\u7d04"
{Constraints} "\u5236\u7d04"
{Columns for 'UPDATE OF'} "'UPDATE OF'\u306e\u5217"
{Confirm Query} "\u30af\u30a8\u30ea\u78ba\u8a8d"
{Search} "\u691c\u7d22"
{Replace} "\u7f6e\u63db\u3048"
{Search Up} "\u4e0a\u691c\u7d22\u2191"
{Search Down} "\u4e0b\u691c\u7d22\u2193"
{Pattern Match} "\u30d1\u30bf\u30fc\u30f3\u30de\u30c3\u30c1"
{None} "\u7121\u3057"
{RegExp} "\u6b63\u898f\u8868\u73fe"
{NoCase} "\u5927\u6587\u5b57\u5c0f\u6587\u5b57\u3092\u533a\u5225\u3057\u306a\u3044"
{Data Editor} "\u30c7\u30fc\u30bf\u7de8\u96c6"
{Unknown Image Format} "\u753b\u50cf\u5f62\u5f0f\u304c\u4e0d\u660e\u3067\u3059"
{General} "\u4e00\u822c"
{SQL Editor} "SQL\u30a8\u30c7\u30a3\u30bf"
{Font Name} "\u30d5\u30a9\u30f3\u30c8\u540d"
{Font} "\u30d5\u30a9\u30f3\u30c8"
{Size} "\u30b5\u30a4\u30ba"
{String Style} "\u6587\u5b57\u30b9\u30bf\u30a4\u30eb"
{Bold} "\u592a\u5b57"
{Underline} "\u4e0b\u7dda"
{Italic} "\u659c\u4f53"
{Uppercase} "\u5927\u6587\u5b57\u306b\u5909\u63db\u3059\u308b"
{Encoding} "\u6587\u5b57\u30b3\u30fc\u30c9"
{Grid Table} "\u30b0\u30ea\u30c3\u30c9\u30c6\u30fc\u30d6\u30eb"
{Use Vi like cursor move} "Vi\u98a8\u30ab\u30fc\u30bd\u30eb\u79fb\u52d5"
{Max History} "\u5c65\u6b74\u4fdd\u5b58\u6570"
{Create/Vacuum 3.x File as 3.x Compatibility Format} "\u5168\u3066\u306e3.x\u3067\u4f7f\u3048\u308b\u5f62\u5f0f\u3067DB\u3092\u4f5c\u6210/Vacuum\u3059\u308b\u3002"
{Version 3.x File Creation} "3.x\u5f62\u5f0f\u306eDB\u4f5c\u6210"
{Page Size} "\u30da\u30fc\u30b8\u30b5\u30a4\u30ba"
{Tcl Script / Shared Library} "Tcl\u30b9\u30af\u30ea\u30d7\u30c8/\u5171\u6709\u30e9\u30a4\u30d6\u30e9\u30ea"
{NOTE:They'll be loaded when db is opened.} "\u6ce8\u610f:\u3053\u308c\u3089\u306fDB\u3092\u958b\u304f\u6642\u306b\u8aad\u307f\u8fbc\u307f\u307e\u3059\u3002"
{Configure entry point of sqlite extension} "SQLite\u62e1\u5f35\u306e\u30a8\u30f3\u30c8\u30ea\u30dd\u30a4\u30f3\u30c8\u306e\u8a2d\u5b9a"
{Enter a entry point name.} "\u30a8\u30f3\u30c8\u30ea\u30dd\u30a4\u30f3\u30c8\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{If the value is empty, default entry point is used for shared library. And if the library is a Tcl script, the value is ignored.} "\u5024\u304c\u7a7a\u306e\u5834\u5408\u306f\u30c7\u30d5\u30a9\u30eb\u30c8\u306e\u30a8\u30f3\u30c8\u30ea\u30dd\u30a4\u30f3\u30c8\u3092\u4f7f\u3044\u307e\u3059\u3002\u307e\u305f\u30e9\u30a4\u30d6\u30e9\u30ea\u304cTcl\u30b9\u30af\u30ea\u30d7\u30c8\u306e\u5834\u5408\u3001\u3053\u306e\u5024\u306f\u4f7f\u7528\u3055\u308c\u307e\u305b\u3093\u3002"
{Export Text Format} "\u30c6\u30ad\u30b9\u30c8\u66f8\u304d\u51fa\u3057"
{Import Text Format} "\u30c6\u30ad\u30b9\u30c8\u8aad\u307f\u8fbc\u307f"
{Source} "\u51fa\u529b\u5143"
{Import to} "\u5165\u529b\u5148"
{New Table} "\u65b0\u898f\u30c6\u30fc\u30d6\u30eb"
{Result Table} "\u7d50\u679c\u30c6\u30fc\u30d6\u30eb"
{Separator} "\u533a\u5207\u308a\u6587\u5b57"
{Comma} "\u30b3\u30f3\u30de"
{Tab} "\u30bf\u30d6"
{Record Type} "\u6539\u884c\u65b9\u5f0f"
{Quote char} "\u30af\u30a9\u30fc\u30c8\u6587\u5b57"
{Double Quote} "\u30c0\u30d6\u30eb\u30af\u30a9\u30fc\u30c8"
{Single Quote} "\u30b7\u30f3\u30b0\u30eb\u30af\u30a9\u30fc\u30c8"
{Auto} "\u81ea\u52d5"
{Output Encoding} "\u51fa\u529b\u5148\u6587\u5b57\u30b3\u30fc\u30c9"
{Input Encoding} "\u5165\u529b\u5143\u6587\u5b57\u30b3\u30fc\u30c9"
{Output Field Names} "\u5217\u540d\u3092\u51fa\u529b\u3059\u308b"
{First Line is Field Names} "\u4e00\u884c\u76ee\u3092\u5217\u540d\u3068\u3057\u3066\u6271\u3046"
{Cache size} "\u30ad\u30e3\u30c3\u30b7\u30e5\u30b5\u30a4\u30ba"
{Default cache size} "\u30c7\u30d5\u30a9\u30eb\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u30b5\u30a4\u30ba"
{Please open a Database file.} "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u3092\u958b\u3044\u3066\u304f\u3060\u3055\u3044"
{The file doesn't exist.} "\u30d5\u30a1\u30a4\u30eb\u304c\u3042\u308a\u307e\u305b\u3093\u3002"
{Failed to open the file.} "\u30d5\u30a1\u30a4\u30eb\u3092\u958b\u3051\u307e\u305b\u3093\u3067\u3057\u305f\u3002"
{The file is not a SQLite file.} "SQLite\u306e\u30d5\u30a1\u30a4\u30eb\u3067\u306f\u3042\u308a\u307e\u305b\u3093\u3002"
{Faied to open the file. The file has zero size.} "\u7a7a\u30d5\u30a1\u30a4\u30eb\u306a\u306e\u3067\u958b\u3051\u307e\u305b\u3093\u3002"
{Failed to attach the db. The db version is not same as current db.} "main\u306e\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u3068\u30d0\u30fc\u30b8\u30e7\u30f3\u304c\u9055\u3046\u306e\u3067\u30a2\u30bf\u30c3\u30c1\u3067\u304d\u307e\u305b\u3093"
{Enter database name.} "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u540d\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044"
{May I detach the database?} "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u3092\u30c7\u30bf\u30c3\u30c1\u3057\u3066\u3044\u3044\u3067\u3059\u304b\uff1f"
{"Please install tclsqlite, if you want to open SQLite2 file."} "SQLite2\u306b\u306f\u5bfe\u5fdc\u3057\u3066\u3044\u307e\u305b\u3093\u3002tclsqlite\u3092\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u3057\u3066\u304f\u3060\u3055\u3044"
{"Please install tclsqlite3, if you want to open SQLite3 file."} "SQLite3\u306b\u306f\u5bfe\u5fdc\u3057\u3066\u3044\u307e\u305b\u3093\u3002tclsqlite3\u3092\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u3057\u3066\u304f\u3060\u3055\u3044"
{Unsupported tclsqlite. I require a tclsqlite compiled for ISO8859 encoding.} "tclsqlite\u304cISO8859\u5bfe\u5fdc\u3067\u30b3\u30f3\u30d1\u30a4\u30eb\u3055\u308c\u3066\u3044\u307e\u305b\u3093\u3002"
{Please select a database.} "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{Please select a table.} "\u30c6\u30fc\u30d6\u30eb\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{Please select tables or views.} "\u30c6\u30fc\u30d6\u30eb\u304b\u30d3\u30e5\u30fc\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{Please select an index.} "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{Please select a trigger.} "\u30c8\u30ea\u30ac\u30fc\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{May I really drop the tables or views?} "\u672c\u5f53\u306b\u30c6\u30fc\u30d6\u30eb/\u30d3\u30e5\u30fc\u3092\u524a\u9664\u3057\u3066\u3082\u3044\u3044\u3067\u3059\u304b\uff1f"
{May I really drop the index?} "\u672c\u5f53\u306b\u30a4\u30f3\u30c7\u30c3\u30af\u30b9\u3092\u524a\u9664\u3057\u3066\u3082\u3044\u3044\u3067\u3059\u304b\uff1f"
{May I really drop the trigger?} "\u672c\u5f53\u306b\u30c8\u30ea\u30ac\u30fc\u3092\u524a\u9664\u3057\u3066\u3082\u3044\u3044\u3067\u3059\u304b\uff1f "
{Failed to drop the tables/views. A table/view with tha same name already exists on other database.} "\u540c\u540d\u306e\u30c6\u30fc\u30d6\u30eb\u304b\u30d3\u30e5\u30fc\u304c\u4ed6\u306e\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u4e0a\u306b\u5b58\u5728\u3057\u3066\u3044\u308b\u306e\u3067\u524a\u9664\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002"
{Failed to create the table. A table with the same name already exists.} "\u540c\u540d\u306e\u30c6\u30fc\u30d6\u30eb\u304b\u30d3\u30e5\u30fc\u304c\u5b58\u5728\u3057\u3066\u3044\u308b\u306e\u3067\u4f5c\u6210\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002"
{More than one rows were updated by trigger.  Data of table may not be synchronized.  Please re-open this table for synchronizing.} "\u8907\u6570\u306e\u884c\u304c\u30c8\u30ea\u30ac\u30fc\u306b\u3088\u3063\u3066\u66f4\u65b0\u3055\u308c\u307e\u3057\u305f\u3002 \u30c6\u30fc\u30d6\u30eb\u306e\u30c7\u30fc\u30bf\u304c\u540c\u671f\u3055\u308c\u3066\u3044\u306a\u3044\u53ef\u80fd\u6027\u304c\u3042\u308a\u307e\u3059\u3002 \u30c7\u30fc\u30bf\u3092\u540c\u671f\u3059\u308b\u305f\u3081\u306b\u30c6\u30fc\u30d6\u30eb\u3092\u958b\u304d\u76f4\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
{You can't modify the table in view mode.} "\u95b2\u89a7\u5c02\u7528\u30e2\u30fc\u30c9\u306a\u306e\u3067\u30c6\u30fc\u30d6\u30eb\u3092\u7de8\u96c6\u3067\u304d\u307e\u305b\u3093\u3002 "
{Failed to import SQL.} "SQL\u306e\u8aad\u307f\u8fbc\u307f\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"
{Note:If import file size was big, we would preview top 50KB of the file.} "\u6ce8\u610f:\u30c7\u30fc\u30bf\u304c\u5927\u304d\u3044\u5834\u5408\u3001\u5148\u982d\u306e50KB\u306e\u307f\u8aad\u307f\u8fbc\u3093\u3067\u30d7\u30ec\u30d3\u30e5\u30fc\u8868\u793a\u3059\u308b\u306e\u3067\u672b\u5c3e\u306e\u30c7\u30fc\u30bf\u304c\u3061\u3087\u3093\u5207\u308c\u3066\u898b\u3048\u308b\u5834\u5408\u304c\u3042\u308a\u307e\u3059\u3002"
{Run vacuum?} "vacuum\u3057\u307e\u3059\u304b\uff1f"
{If this DB was big, vacuum will take a long time.} "\u3082\u3057\u3053\u306eDB\u304c\u5927\u304d\u3051\u308c\u3070vacuum\u306f\u6642\u9593\u304c\u304b\u304b\u308a\u307e\u3059\u3002"
{Cache size reverts to the default value when this DB is closed. But Default cache size value is stored in DB. Temp DB's cache size and default cache size aren't stored in DB.} "\u30ad\u30e3\u30c3\u30b7\u30e5\u30b5\u30a4\u30ba\u306e\u5024\u306f\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u3092\u9589\u3058\u308b\u3068\u521d\u671f\u5316\u3055\u308c\u307e\u3059\u304c\u3001\u30c7\u30d5\u30a9\u30eb\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u30b5\u30a4\u30ba\u306e\u5024\u306f\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u306b\u683c\u7d0d\u3055\u308c\u6c38\u7d9a\u3057\u307e\u3059\u3002Temp\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u306e\u5834\u5408\u306f\u3069\u3061\u3089\u306e\u5024\u3082\u6c38\u7d9a\u3057\u307e\u305b\u3093\u3002"
}


#
# Icon Images
#
set img(arrowDown) [image create photo ::image::arrowDown -data {
R0lGODlhEAAQALMAAGB0wZOgzMzMzHuJvaSu04SRw5unz46byamy1ZejzomW
xqCr0f///wAAAAAAAAAAACH5BAUUAAwALAAAAAAQABAAAARFkMlJq70Y6K3F
BUgoAp4FEGhKfkvrrqYhz7C0JXiObx4Q/MDgDwY4GI/IGgOgaDqbSlthSo1O
AIPswErBbksYG/cCpkQAADs=
}]

set img(arrowLeft) [image create photo ::image::arrowLeft -data {
R0lGODlhEAAQALMAAF5zwZejzomWxszMzHuJvaSu05OgzJunz46byaCr0WB0
wamy1f///wAAAAAAAAAAACH5BAUUAAwALAAAAAAQABAAAAROkMlJq5VAXauW
GtukFMUXMkqiJkrbVsAhz7OJBXiu24Dh/0ADD0EsGhG2geJ4tDGUgqhU6nwq
CFiCSwEAUQYA7GdAJlvAWu8J3D191ZMIADs=
}]

set img(arrowRight) [image create photo ::image::arrowRight -data {
R0lGODlhEAAQALMAAF5zwZOgzImWxszMzHuJvZunz6Su046byZejzqCr0WB0
wamy1f///wAAAAAAAAAAACH5BAUUAAwALAAAAAAQABAAAARMkMlJq5UK3KuW
2pViGJ+lnGeiJiWlFHAcay5i3ziNBXzvBzrFYUgsHhQDjNGInCgE0Gi0OQGg
FIQsgToZeL3YbRLEwHJBAMCYzPBeIgA7
}]

set img(arrowUp) [image create photo ::image::arrowUp -data {
R0lGODlhEAAQALMAAGB0wZOgzMzMzHuJvaCr0YSRw5ejzqSu046byZunz4mW
xqmy1f///wAAAAAAAAAAACH5BAUUAAwALAAAAAAQABAAAARJkMlJq5UAXAvW
0ht2jAd4AUSqmhSQvPDLMoBh37gJBHzv86AMAEEsEoUCiUAAUDifgOWyVaha
o6eBdovlbLlJjnAcvkiloTQjAgA7
}]

set img(checkAutoincr) [image create photo ::image::checkAutoincr -data {
R0lGODlhEAAQAMQAAPjH3PfA2czMzPa71fS10fS10PW00fOwzPKxzPOuzfKp
yvKnx++mx/CjxPCjxe+ixPChxO+Yv+6XvO2WveyRuOyPtuuMtOuIsumEr+h7
qed2pv///wAAAAAAAAAAAAAAACH5BAUUABsALAAAAAAQABAAAAVI4CaOZGme
aKqWwDoOQeESBMII6nE0jYCjh4VE4iNFSAqFhFIcXSLHjQJSqTRHmmfE4bhg
riNBFnrJgEniy0VzLonZv1XbdQoBADs=
}]

set img(checkNNull) [image create photo ::image::checkNNull -data {
R0lGODlhEAAQAMQAAMzMzLzG7bTA66676am046i15aWz5qOv4KKv4Z6t4pqp
3pmn2Jem3pOj2pOh2pSi1o+f2oeY1oaW0YWV04CQz3yNz3mKzHWHzHCCx2h7
xGB0wQAAAAAAAAAAAAAAAAAAACH5BAUUABsALAAAAAAQABAAAAVI4CaOZGme
aKqWwToOAuEWxbEAKoI4DoCjCMZk4iNFSArFhFIcXSLHjQJSsTRHmmekAblg
riNAFnrJgEniy0VzLonZv1XbdQoBADs=
}]

set img(checkPkey) [image create photo ::image::checkPkey -data {
R0lGODlhEAAQAMQAAMzMzP2+xP24v/yzuvyrs/ypsfyjrPyhqvucpfuao/uW
oPuUnvqRmvuQmvuIk/qFkPp/ivp7h/p3g/pzgPlue/lkcvlca////wAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUABcALAAAAAAQABAAAAVI4CWOZGme
aKqWwToOQuEShKEAqmEsDICjhsTj4SM5SAjEA1IcTRzHC6IRkTRHlqdj0ZhQ
riNAFjqpgEniycRyLonZv1XbdQoBADs=
}]

set img(checkUnique) [image create photo ::image::checkUnique -data {
R0lGODlhEAAQAJEAAGLgRczMzP///wAAACH5BAUUAAIALAAAAAAQABAAAAIm
lI+py+0J3gFUUhrcBSGz3SHRdIXkCHrnxqlHwLYMXLoKbT655BQAOw==
}]

set img(columnEdit) [image create photo ::image::columnEdit -data {
R0lGODlhEAAQANUAAEdhMszMzDdt+yZLrY+l/////3XYFpzWca2trdHX/37V
M3OiVMvycWyZT015/3uEsLK//531V3jwAHqzUG+uPNTU1KbyQrTwfXfnCH3x
DW6kRYbAY3mqWKbgeXntBXjmDnbfD6zne4bCZQAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUACMALAAAAAAQABAAAAZ2
wJFwSCwSCY6kMkkwLp8CCoXoKFivVoqFMRlWsVlJJrLwJiBoNEUiwXCoYErG
A3rDrxoJnRMoOs4QeR8GHAN9dw0SHwoLCAKHXgUiFx8bjRWPfgkXIQcNDxAJ
mXAAHQ0IV6ORCKxYqkIQArKzshCQQwG5urtCQQA7
}]

set img(columnNew) [image create photo ::image::columnNew -data {
R0lGODlhEAAQAMQAAM50DDdt+////8zMzI+l//HXA/z3AdJ/C/Pckua6R9HX
/015/+i9BrK//9ybCdmUCfXjAtB7C9WICtaTFt6jCPnvAf//ANKBC9SMEwAA
AAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUABkALAAAAAAQABAAAAVlYCaOZAaU
IrGsqwC8KNsI7kHB5EIjr1RUjgguo9s9KpYkI5JTNGYYRtLgKBVpAAqEATnl
aLTEJUIeEp2zF3jiFV3BtIDACocHBlb086m40+txeF+AAn4kDQGJiokNgiQD
kJGSIiEAOw==
}]

set img(columnNormal) [image create photo ::image::columnNormal -data {
R0lGODlhEAAQAKIAADdt+////8zMzLK//015/4+l/9HX/wAAACH5BAUUAAcA
LAAAAAAQABAAAAM+eLrcXCTKWNy8QDQSuu/Z9n0hQxhDqhrlwo2gZsLxhqop
K7t00Cov2O9wwuWGwRGyNxwAntDnYLcQWK9YRQIAOw==
}]

set img(db) [image create photo ::image::db -data {
R0lGODlhEAAQANUAAJuWGczMzIvVBLTyU9n+gKqnIpfZHoWzDZPYFsb8Y+v9
va37IIrGCeD8mZfzCKTqN8j+SZTvCIu3GYvnAKOcE5LKGtP7bZetF5vkJ7r4
V4rfApz1DqKdIeL8oa2nKZryDqTsNqSdGp2aHLb3U465Hp+rHP///wAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUACYALAAAAAAQABAAAAZ1
QJNwSCwOKaGkchkaehQNQmI0AoEwhgqJEzB5OhbIwuGYTDQCxkHU/YbHHzNa
DegW7vj8/VI3FaBSGQNWGAgVEn0FYGIbEXJpB4mLC42PdHZ6mSWJgAmChIaI
dpOVZ5Aho2+lcweoQgVMsQFdJrO2t7dGukRBADs=
}]

set img(dummy16x16) [image create photo ::image::dummy16x16 -data {
R0lGODlhEAAQAIAAAAAAAP///yH5BAUUAAEALAAAAAAQABAAAAIOjI+py+0P
o5y02ouzPgUAOw==
}]

set img(folder) [image create photo ::image::folder -data {
R0lGODlhEAAQAMQAALOBG//MZszMzOSxTP/////mgf/ge+68V//0ju+9Uv/U
b///mfbDXue3UfbGYLWEIf/rhf/3kf///wAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUABIALAAAAAAQABAAAAVVoCSOZGme
J6AC6AgQMMGi72Iv8qq6EeL7sRhAIAFACsiksjAsGp7Q6LMJUFivWCs1wO16
uVSGeCx2mB3Ug3rNVlMH8EFjTp83i7q8SkCU8P+AgC0oIQA7
}]

set img(key) [image create photo ::image::key -data {
R0lGODlhBwAKALMAAKR1EdqqDP/iZbacYbuLA//zg+fCTP//zPPIPqx8Dv3o
dcOSAP///wAAAAAAAAAAACH5BAUUAAwALAAAAAAHAAoAAAQjcKxFSJKnFCHS
KgGwIB5nICSTJMbKvEv7wu1EK+GNAHAC/BEAOw==
}]

set img(lock) [image create photo ::image::lock -data {
R0lGODlhCQAJAMQAALSDCNqqDPfidbacYfDXZufCTKR1Ef//zPPIPrKKK7uL
A9ejCf3ZW/PNS6x8Dv/zg//iZfPZa/HQVPzVUsOSAP/laP3odffaZfzRSf//
/wAAAAAAAAAAAAAAAAAAAAAAACH5BAUUABkALAAAAAAJAAkAAAU0YJY5JCmO
wzk4InuOGaXMCgCQSnUcEcQEDgXkIEhIJgGD8EEwYpIKhuVSaCAWStvNYeiG
AAA7
}]

set img(nullValue) [image create photo ::image::nullValue -data {
R0lGODlhEAAQALMAAP///8zMzMLJ6bjB5K+54Jqn3JGf14eW036Oz3SGymt+
xmJ2wgAAAAAAAAAAAAAAACH5BAUUAAwALAAAAAAQABAAAARBkMlJq70467y6
/yCzFGRplkuwAAbrtgacrgJw2PhxHzPgIwCgUNhLAIw+5MBHUCme0Ch0Bqp6
AoyAdsvtbr7gbwQAOw==
}]

set img(primary_key) [image create photo ::image::primary_key -data {
R0lGODlhBwAKALMAAKR1EdqqDP/iZbacYbuLA//zg+fCTP//zPPIPqx8Dv3o
dcOSAP///wAAAAAAAAAAACH5BAUUAAwALAAAAAAHAAoAAAQjcKxFSJKnFCHS
KgGwIB5nICSTJMbKvEv7wu1EK+GNAHAC/BEAOw==
}]

set img(table) [image create photo ::image::table -data {
R0lGODlhEAAQAKIAADdt+9HX/7K//////8zMzE15/4+l/////yH5BAUUAAcA
LAAAAAAQABAAAANMeLrcaiXKGY2iOAJyyvjBB4pbJwrigH5lEQjCC8dw0J7p
OtyfrpIcV21Gi/F+OFYw2QN2ZLKhcdlkHkMi7I4jAHi/YK+AcyCYz2h0AgA7
}]

set img(view) [image create photo ::image::view -data {
R0lGODlhEAAQAMQAAC6Tk5SUlADBwWZmZv///wDl5SioqHvT04e5uUeMjADT
0wC1tQD//7S0tCO1tVF/fxu9vZ7JySienhm0tJmZmQDe3iutrUeUlADFxS65
uQDv7y6dnTqUlC6lpf///wAAACH5BAUUAB4ALAAAAAAQABAAAAVWoCeOZGme
qDek5UCtJiwOjRq/c22TtN2fv1UgcEkMBsOiTtQgEBiFRPNZ6JAQVMUkkjWQ
MtBCQQLRVCoCCamjwCwsnoMDI1gASpsN4CHi5O8sgYKDJiEAOw==
}]

set img(writable) [image create photo ::image::writable -data {
R0lGODlhCQAJALMAAARao1y9eGZmZp/w/DeExpCQkFvTphFzo3Ttl3nPuG3g
j2XbsHn3nv///wAAAAAAACH5BAUUAA0ALAAAAAAJAAkAAAQesMl5zqSDWAw0
14V0DN0RNmN5Jos6MYhhXoISnFMEADs=
}]



#
# Some fix of theme look&feel
#
#improving look and feel of ttk and tk.
#just source this
namespace eval improvettk {}

proc improvettk::improve_themes {} {

	set tile_version [package require tile]
	if {[string match 0.7.* $tile_version]} {
		set tile_ns tile
		set style style
	} else {
		set tile_ns ttk
		set style ttk::style
	}

	foreach theme [$style theme names] {

		switch -exact -- $theme {
			"winnative" {
				$style theme settings $theme {
					$style map TButton \
			 		-foreground [list !disabled SystemButtonText disabled SystemDisabledText] \
					;
					$style map TCheckbutton \
			 		-foreground [list !disabled SystemWindowText disabled SystemDisabledText] \
					;
					$style map TRadiobutton \
			 		-foreground [list !disabled SystemWindowText disabled SystemDisabledText] \
					;
					$style configure TEntry \
					-foreground SystemWindowText \
					-fieldforeground SystemWindowText \
					-fieldbackground SystemWindow \
					;
					$style map TCombobox \
					-fieldbackground [list !disabled SystemWindow disabled SystemButtonFace] \
					-foreground	[list {readonly focus} SystemHighlightText disabled SystemDisabledText] \
					;
					$style map TMenubutton \
			 		-foreground [list !disabled systemButtonText disabled SystemDisabledText] \
			 		-arrowcolor [list !disabled systemButtonText disabled SystemDisabledText] \
					-embossed [list disabled 1] \
					;
					$style map TNotebook.Tab \
			 		-foreground [list !disabled systemButtonText disabled SystemDisabledText] \
					;
				}
			}
			"xpnative" {
					$style map TCombobox \
					-foreground	[list {readonly focus} SystemHighlightText disabled SystemGrayText] \
					;
			}
			"clam" {
				upvar ${tile_ns}::theme::${theme}::colors colors
				$style theme settings $theme {
					$style map TCheckbutton \
			 		-indicatorforeground [list disabled $colors(-disabledfg)] \
					;
					$style map TRadiobutton \
			 		-indicatorforeground [list disabled $colors(-disabledfg)] \
					;
					$style map TEntry \
					-fieldbackground [list readonly $colors(-frame) disabled $colors(-frame)] \
					;
					$style map TCombobox \
					-fieldbackground [list readonly white disabled $colors(-frame)] \
					-foreground [list disabled $colors(-disabledfg)] \
					-background [list disabled $colors(-frame) active $colors(-lighter) pressed $colors(-lighter)] \
			 		-arrowcolor [list disabled $colors(-disabledfg)] \
					;
					$style map TMenubutton \
			 		-arrowcolor [list disabled $colors(-disabledfg)] \
					;
					$style map TScrollbar \
			 		-arrowcolor [list disabled $colors(-disabledfg)] \
					;
				}
			}
		}
	}
	return
}
improvettk::improve_themes


proc improvettk::get_ttk_options {theme class option {state ""}} {
	set tile_version [package require tile]
	if {[string match 0.7.* $tile_version]} {
		set tile_ns tile
		set style style
	} else {
		set tile_ns ttk
		set style ttk::style
	}
	set val [$style theme settings $theme {$style lookup $class $option $state}]
	if {$val ne ""} {return $val}
	set val [$style theme settings $theme {$style lookup . $option $state}]
	if {$val ne ""} {return $val}
	set val [$style theme settings default {$style lookup $class $option $state}]
	if {$val ne ""} {return $val}
	set val [$style theme settings default {$style lookup . $option $state}]
	if {$val ne ""} {return $val}
	return ""
}

proc improvettk::set_tk_color_as_tile {} {
	set tile_version [package require tile]
	if {[string match 0.7.* $tile_version]} {
		set tile_ns tile
		set style style
	} else {
		set tile_ns ttk
		set style ttk::style
	}
	
	set theme [set ${tile_ns}::currentTheme]
	set bg       [get_ttk_options $theme . -background]
	set fg       [get_ttk_options $theme . -foreground]
#    set activebg [get_ttk_options $theme . -background active]
#    set activefg [get_ttk_options $theme . -foreground active]
	set selectbg [get_ttk_options $theme . -selectbackground focus]
	set selectfg [get_ttk_options $theme . -selectforeground focus]
	set disabled [get_ttk_options $theme . -foreground disabled]
	set buttonfg_normal   [get_ttk_options $theme TButton -foreground]
	set buttonbg_normal   [get_ttk_options $theme TButton -background]
	set buttonfg_disabled [get_ttk_options $theme TButton -foreground disabled]
	set buttonbg_disabled [get_ttk_options $theme TButton -background disabled]
	set buttonfg_readonly [get_ttk_options $theme TButton -foreground readonly]
	set buttonbg_readonly [get_ttk_options $theme TButton -background readonly]
	set buttonfg_active   [get_ttk_options $theme TButton -foreground pressed]
	set buttonbg_active   [get_ttk_options $theme TButton -background pressed]
	set fieldfg_normal   [get_ttk_options $theme TEntry -foreground]
	set fieldbg_normal   [get_ttk_options $theme TEntry -fieldbackground]
	set fieldfg_disabled [get_ttk_options $theme TEntry -foreground disabled]
	set fieldbg_disabled [get_ttk_options $theme TEntry -fieldbackground disabled]
	set fieldfg_readonly [get_ttk_options $theme TEntry -foreground readonly]
	set fieldbg_readonly [get_ttk_options $theme TEntry -fieldbackground readonly]
	set comboffg_normal   [get_ttk_options $theme TCombobox -foreground]
	set combofbg_normal   [get_ttk_options $theme TEntry    -fieldbackground]
	set comboffg_disabled [get_ttk_options $theme TCombobox -foreground disabled]
	set combofbg_disabled [get_ttk_options $theme TCombobox -fieldbackground disabled]
	set comboffg_readonly [get_ttk_options $theme TCombobox -foreground readonly]
	set combofbg_readonly [get_ttk_options $theme TCombobox -fieldbackground readonly]

	option add *Button.background $buttonbg_normal
	option add *Button.foreground $buttonfg_normal
	option add *Button.activeBackground $buttonbg_active
	option add *Button.activeForeground $buttonfg_active
	option add *Button.disabledForeground $buttonfg_disabled

	option add *Frame.background $bg
	option add *Frame.foreground $fg

	option add *Listbox.background $fieldbg_normal
	option add *Listbox.foreground $fieldfg_normal
	option add *Listbox.selectBackground $selectbg
	option add *Listbox.selectForeground $selectfg
	option add *Listbox.activeStyle none

	option add *Text.background $fieldbg_normal
	option add *Text.foreground $fieldfg_normal
	option add *Text.selectBackground $selectbg
	option add *Text.selectForeground $selectfg

	option add *Entry.background $fieldbg_normal
	option add *Entry.foreground $fieldfg_normal
	option add *Entry.selectBackground $selectbg
	option add *Entry.selectForeground $selectfg
	option add *Entry.disabledBackground $fieldbg_disabled
	option add *Entry.disabledForeground $fieldfg_disabled
	option add *Entry.readonlyBackground $fieldbg_readonly
 	
	option add *Menu.tearOff 0
	if {$theme ne "winnative" && $theme ne "xpnative"} {
	option add *Menu.background $bg
	option add *Menu.foreground $fg
	option add *Menu.activeBackground $selectbg
	option add *Menu.activeForeground $selectfg
	option add *Menu.disabledForeground $disabled
	}

	option add *Label.takeFocus 0

	option add *Spinbox.background $combofbg_normal
	option add *Spinbox.foreground $comboffg_normal
	option add *Spinbox.readonlyBackground $combofbg_readonly
	option add *Spinbox.disabledBackground $combofbg_disabled
	option add *Spinbox.disabledForeground $comboffg_disabled
	option add *Spinbox.selectForeground $selectfg
	option add *Spinbox.selectBackground $selectbg
 
}
improvettk::set_tk_color_as_tile



# wish on linux has "ucs-2be" and "X11ControlChars".
# But wish on windows doesn't have it. so we remove these.
set ENCODING_NAMES [string map {"ucs-2be" "" "X11ControlChars" ""} \
		[lsort -dictionary [encoding names]]]
set SYSENCODING [encoding system]

# Options
# usesession   if not 0, save current window positions.
# appname      currently it is used by window title only.
# enable_encoding  encoding names list.
array set pref [list \
	debug		0 \
	usesession	1 \
	appname		TkSQLite \
	sysencoding $SYSENCODING \
	enable_encoding	$ENCODING_NAMES \
	recent_file [list] \
	open_file "" \
	init_dir  "" \
	openTypeSqlite	[list {"SQLite Files" {.db .db2 .db3 .sdb .sqlite .sqlite3}} {"All Files" *}] \
	openTypeSql		[list {"SQL Files" {.sql}} {"All Files" *}] \
	openTypeText	[list {"Text Files" {.txt .csv .tab .tsv .dat}} {"All Files" *}] \
]

# sessionfile
if {$tcl_platform(platform) eq "windows"} {
	if {[array names env APPDATA] ne ""} {
		set pref(sessionfile) [file join [file normalize $env(APPDATA)] tksqlite .tksqlite]
	} else {
		set pref(sessionfile) [file join [file normalize $env(HOME)] .tksqlite]
	}
} else {
	set pref(sessionfile) [file join [file normalize $env(HOME)] .tksqlite]
}

set pref(openTypeImageNoImg) {{"Image Files" .gif} {"GIF" .gif} {"All Files" *}}
	
set pref(openTypeImageWithImg) {
	{"Image Files" {.gif .jpg .jpeg .png .bmp .ico .pcx .ppm .sgi .sun .tga .tif .tiff .xbm .xpm}}
	{"GIF" .gif}
	{"JPEG" {.jpg .jpeg}}
	{"PNG" .png}
	{"BMP" .bmp}
	{"ICO" .ico}
	{"Paintbrush" .pcx}
	{"Portable Pixmap" .ppm}
	{"SGI" .sgi}
	{"SUN" .sun}
	{"TGA" .tga}
	{"TIFF" {.tif .tiff}}
	{"X Windows Bitmap" .xbm}
	{"X Windows Pixmap" .xpm}
	{"All Files" *}
}

set pref(openTypeExtension) [list \
	{"Tcl Script/Shared Library" {.tcl .dll .so .dylib}} \
	{"Shared Library" {.dll .so .dylib}} \
	{"Tcl Script" .tcl} \
	{"All Files" *}]

set pref(tile_version) [join [lrange [split $tile_version .] 0 1] .]
set pref(tile_selectbg) [style theme settings default {style configure . -selectbackground}]
set pref(tile_selectfg) [style theme settings default {style configure . -selectforeground}]
set pref(tile_bg) [style theme settings $tile_currentTheme {style configure . -background}]
set pref(tile_fg) [style theme settings $tile_currentTheme {style configure . -foreground}]
if {$pref(tile_bg) eq ""} {
set pref(tile_bg) [style theme settings default {style configure . -background}]
set pref(tile_fg) [style theme settings default {style configure . -foreground}]
}

# modify class and reset look & feel
if {$pref(tile_version) >= 0.8} {
	foreach _treeitem {Row Cell Item} {
		style map $_treeitem \
			-background [list selected $pref(tile_selectbg) alternate #EFEFEF] \
			-foreground [list selected $pref(tile_selectfg)]
	}
}
option add *Listbox.selectBackground $pref(tile_selectbg)
option add *Listbox.selectForeground $pref(tile_selectfg)
option add *Text.background white
option add *Text.selectBackground $pref(tile_selectbg)
option add *Text.selectForeground $pref(tile_selectfg)
option add *Text.inactiveSelectBackground $pref(tile_selectbg)
option add *Text.inactiveSelectForeground $pref(tile_selectfg)
option add *Listbox.background white
option add *Listbox.activeStyle none
option add *Menu.tearOff 0
option add *Label.takeFocus 0
option add *Button.font      TkDefaultFont
option add *Checkbutton.font TkDefaultFont
option add *Entry.font       TkTextFont
option add *Label.font       TkDefaultFont
option add *Labelframe.font  TkDefaultFont
option add *Listbox.font     TkTextFont
;#option add *Menu.font        TkDefaultFont
option add *Menubutton.font  TkDefaultFont
option add *Radiobutton.font TkDefaultFont
option add *Scale.font       TkDefaultFont
option add *Spinbox.font     TkTextFont
option add *Text.font        TkTextFont

if {[tk windowingsystem] eq "aqua"} {
	set pref(modifier) Command
	set pref(shortmod) Command
} else {
	set pref(modifier) Control
	set pref(shortmod) Ctrl
}

bind Text <$::pref(modifier)-d> {
	if {[llength [%W tag range sel]]} {
		%W delete sel.first sel.last
	}
}
bind Text <$::pref(modifier)-a> {
	%W tag add sel 1.0 end
}
foreach class {TEntry TCombobox} {
	bind $class <$::pref(modifier)-a> {
		%W selection range 0 end
	}
	bind $class <$::pref(modifier)-d> {
		if {[%W selection present]} {
			%W delete sel.first sel.last
		}
	}
}

switch -exact -- [tk windowingsystem] {
	x11 {
		font configure TkHeadingFont -weight normal
		option add *Menu.activeBackground $pref(tile_selectbg)
		option add *Menu.activeForeground $pref(tile_selectfg)
		option add *Menu.font  TkTextFont
		option add *Menu.activeBorderWidth 0
		option add *Menu.borderWidth 1
		option add *.cursor {}
	}
}
foreach _class {TCombobox TEntry} {
	style configure $_class \
		-selectforeground $pref(tile_selectfg) \
		-selectbackground $pref(tile_selectbg) \
		;
}

switch -exact -- $tile_currentTheme {
	aqua {
		rename ttk::scrollbar ttk::_scrollbar
		interp alias {} ::ttk::scrollbar {} ::scrollbar
	}
	clam {
		style map TEntry \
		-fieldbackground [list readonly $pref(tile_bg) disabled $pref(tile_bg)] \
		;
		style map TCombobox \
		-fieldbackground [list readonly white disabled $pref(tile_bg)] \
		-foreground [list disabled #999999] \
		-arrowcolor [list disabled #999999] \
		;
	}
	xpnative -
	winnative {
		style map TCombobox \
		-fieldbackground [list disabled SystemButtonFace !disabled white] \
		-selectbackground [list !focus $pref(tile_selectbg)] \
		-selectforeground [list !focus $pref(tile_selectfg)] \
		-foreground	[list {readonly focus} SystemHighlightText disabled SystemGrayText] \
		-focusfill	[list {readonly focus} $pref(tile_selectbg)] \
		;
	}
}

# Toolbutton for Tile button
style configure Toolbutton -padding 1

# Checkbutton for treectrl. This has indicator only.
# We can not create own state in ttk, so we must change two styles.
style layout CheckInTree { Checkbutton.indicator }
style configure CheckInTree -padding 0 -background white
style layout CheckInTreeSelected { Checkbutton.indicator }
style configure CheckInTreeSelected -padding 0 -background $pref(tile_selectbg)


# Do not change below
# ---------------------------------------------------------------------

#
# FS Dialogs
#
if {$tcl_platform(platform) eq "unix"} {
# Copyright (C) Schelte Bron.  Freely redistributable.

proc ttk::messageBox {args} {
	return [eval ::Message::show $args]
}

interp alias {} ttk_messageBox {} ::ttk::messageBox

namespace eval ::ttk::dialog {}
namespace eval ::ttk::dialog::file {
	variable sort name hidden 1 sepfolders 1 foldersfirst 1
	variable details 0 reverse 0 filetype none
	variable dirlist "" filelist ""
}
namespace eval ::ttk::dialog::image {}

# Images for the configuration menu

image create photo ::ttk::dialog::image::blank16 -height 16 -width 16

image create photo ::ttk::dialog::image::tick16 -data {
R0lGODlhEAAQAMIAAExOTFRSVPz+/AQCBP///////////////yH5BAEKAAQALAAAAAAQABAA
AAM4CAHcvkEAQqu18uqat+4eFoTEwE3eYFLCWK2lelqyChMtbd84+sqX3IXH8pFwrmNPyRI4
n9CoIAEAOw==}

image create photo ::ttk::dialog::image::radio16 -data {
R0lGODlhEAAQAMIAAJyZi////83OxQAAAP///////////////yH5BAEKAAEALAAAAAAQABAA
AAMtGLrc/jCAOaNsAGYn3A5DuHTMFp4KuZjnkGJK6waq8qEvzGlNzQlAn2VILC4SADs=}

# Images for ttk::getOpenFile, ttk::getSaveFile, ttk::getAppendFile

image create photo ::ttk::dialog::image::next -data {
R0lGODlhFgAWAMYAADt1BDpzBFiJKb7ZpGaVOTx2A8HcqbfVm3ShSjt0BDp1BDx3Bb/apYe7
V7DSkIOtWzt0A8Dbpr/apL7ao7zZoXu0RXy0R6bMgo23Zz12CbzZoH+2Sn61SX21R3qzRHiy
QnaxPnOvOnCuNpjFb5e/cUV8ELnXnHiyQXaxP3WwPXCtNm2sMmqqLWaoKIm8WJ3FeEuBGLXV
l2+tNGGlIWanJ2urLWutLmqtK2irJ2SpIl+lHJ/GeFaKIjt1A6jNhU+aB06aBk+cBlKhCFWl
CViqDF6uEmCvFWGtFl2qE3e2Op3HdVWLIjt2BKPLflSjCFipClyvDF6zDWC2Dl+0DYTER5zK
cEqDFjt3A1eoClywDGG3DmW9EGfBEWnCE5XTWZjJZ0R9D6TLfqbPf6nUgazYgq/cg2nDEXPM
GqPfaY7DWj53CTlzBD13Ba7bg3HGH6fecn+0SqbWdmufOjhwBKTPelqNKTNmAk6DHi9dAzdu
A////////////////////////yH5BAEKAH8ALAAAAAAWABYAAAfGgH+Cg4SFhoeIiYgAio0B
Ao2JAQMEBZGGAQYHCAmNCgGgoAsMDQ4PEIoBEasREhMUFRYXGBmSGhsbHB0eHyAhIiMkJYgB
JifHKCkhKissLS4vMIcBMTItMzM0NTY3ODk6Jzs9mD4/QEBBQkNERUZHSElKTJhN50FOT1BR
UlJTVFVXptUDIgRLFi1buHTx8gUMsSZNwogZQ6aMmTNo0qhJtCYUKDZt3LyB0+mSoABk4siZ
Y3JQADp17LR0eQfPzEF5burcKSgQADs=}

image create photo ::ttk::dialog::image::nextbw -data {
R0lGODlhFgAWAOcAAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0N
DQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8f
HyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDEx
MTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkND
Q0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVV
VVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdn
Z2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5
eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouL
i4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2d
nZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+v
r7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHB
wcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT
09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl
5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf3
9/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///yH5BAEKAP8ALAAAAAAWABYAAAjUAP8JHEiwoMGD
CBMivKKwYZU3DRNWWcaHYcSCVZwVS2SloZUqIEFiYQYK2KWOEpupbLZsmTJLl3CFwiJRWaZM
mDBVoiQJkiNXqr4grHKMklFJkSA1WpTIUChYZQ5WIdbIkCBBhRItUoSoECBKsSwSrJJrjhw5
dPDsAUTIUCFBlcIarGLrLB09fwgdSpQI0ShZNOfWlYPHDyFFjyRRkvVKqFRbkHP1CkaMUidg
p7JIDAkyyzBNwTChvPivSrBehKaQHlgFl5wlq1mfKRJ7YJTauHMLDAgAOw==}

image create photo ::ttk::dialog::image::previous -data {
R0lGODlhFgAWAOcAADp0BFSIJTx1Bzp0A2KSNLrWnz93Czt1BHGeRbXUmL/apTx0BH6qVa/R
joS5UrzZoEF7CzpzBD13CIu2Y6TLf3iyQniyQbnXnbzZob7ao7/apMDbpj92CkR7D5S8bJbD
a22sMW+tNHKvOXaxPnqzRH21R361SX+2SrvYn0mAFprDdIe6VWOmI2aoKGqqLW2sMnCtNnOv
OnWwPXaxP7jWmj52CTt1A1SIIJvEdHWxPlqhF16jHGGlIWSnJWmrK2uvLGqwKGevI2uvKXKy
NrTVlT11CDt3A1SKIJrEcVOdDVWeEFSeD1ekD1enC1mrCluuC1ywDFqqC6rThEmCFZXAbE6a
BlKgB1enCV+0DWK4DmS7D2O7D1+zDajUfkJ5DYy5YYa7U1elDFqsC2jBEWvGEmrFEmfBEWO6
D6rXfzx1CDx2B4GwU5TGY2GxFGC2Dq7dgLLhhLXmhrTlha/dg63Zgjx2CDpyA3WmRZ3Ob2m5
HK3bgEF9CTtzBDduA2aYNqHQdazYgTNlAleLJaPOeS1ZA0yBGzx0Bv//////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////yH5BAEKAP8ALAAAAAAWABYAAAjgAP8JHEiwoMGD
CBMq/AdgYcIAAhwaHECggAGJBA8gSKDgIsZ/Cxg0cPAAQoSTJw8klDCBQgULFzBk0LChJgeE
HTx8ABFCxIgKJEqYOIHipsEUKlawaOHiBYwYMmZYsECjhkEbOHLo2MGjh48fQIIEETKESBGD
RpDASKJkCZMmTp5AgfIkipSzBgFQIVHFypUnWLJo2ZKFSxe8Br18ARNGDBYtY8iUMXMGTZqE
atawaePmDZw4cuDMoVNHoZ07ePLo2YPyJJ+Fffz8AVT6o8BAggbVtv2PUCFDvAn2CU7cdkAA
Ow==}

image create photo ::ttk::dialog::image::previousbw -data {
R0lGODlhFgAWAOcAAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0N
DQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8f
HyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDEx
MTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkND
Q0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVV
VVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdn
Z2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5
eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouL
i4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2d
nZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+v
r7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHB
wcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT
09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl
5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf3
9/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///yH5BAEKAP8ALAAAAAAWABYAAAjXAP8JHEiwoMGD
CBMq/GdlYcI2VxwatJLnmBaJBK8YIsbsIkaGk351UtalikmTERFm+WSLEqVjypYta0YzC0Iv
p1YtavRIEqVKmDBlSmbT4BhXnwYZSrQTUiSflIwVzehKEp8/ggglYsRIkaJFkYhhMYjFVSM7
ePDw6QNoECFCgwD5GjsxVSU5d/oMQrSz0aJDvega5BLqE59AiBpJsmRJUqNfKQ9iucTqUCJi
yJgtQ1ZMmOCDVBjRejTMy8mTC6P4uRXsM8YlcG65xiikTOSPA6Pg3s1bYEAAOw==}

image create photo ::ttk::dialog::image::up -data {
R0lGODlhFgAWAOcAADx2Azx2BFWLIlWNIjx1A0uBGJzFdZrFckuDF0V8EJzDdnKvOm+tNZbB
bUR7Dz52CZa/cIW5UlqhGFaeEXmzQ467ZD14CIy2ZZTCaWOmI16jHFmgFlSdDoO4UIKyVTt1
Azp2BIKsWaLKfWysMWaoKGGlIVyiGlSeD0+bCI2+X3anSDt2BHShSa3RjXexQG+tNGusLWir
J2GoHFOhCVGgB1ahDpXDamiaOWSUN7XVmIS4UXm1QXe1O3O0NG6zLFyqEVeoClenCVamCV6o
F5zIcluOKViKKLvXoL/bpb7bo73coH27QXm8OmWzGVywDFyvDKrVganTgKjRgKDLeU+FHzt1
BDpzBD14BcDeooLBRXK7KmC2DmG4DmK4DmG3DqzZgj55BcLhpYfHSma6FGW9EGe/EGfAEWa/
EK/cg8TjpnzDOGnDEWvHEmzIE2vGErDfhMbkqXTBKW/MFHHQFW3KE7HghGy9Hma+EGrEEm3J
E27LFGzHE7HfhMXjqa/aha7bg7Deg6/dg///////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////yH5BAEKAP8ALAAAAAAWABYAAAjWAP8JHEiwoMGD
CBMqXMiw4UEAARwWLGDgAAGJAhMoWMCggQOJDyBEkDCBQgULDQFcwJBBwwYOHTx8WAgihIgR
JEqYOIEihYoVCQGwaOHiBYwYMmbQqGHjxsWDOHLo2MGjh48fQIIIGUKkyEEjR5AkUbKESRMn
Tp5AiSJlCpWCVazIvYIli5YtXLp4+QJGrhUQCK2EETOGTBkzZ9BYYWgljRoya9i0cfNm8UIr
cOKccSNnDp06lhVitnMHTx49e/iETmilj58/gPjU4RNodWC/uOVi3L07IAA7}

image create photo ::ttk::dialog::image::upbw -data {
R0lGODlhFgAWAOcAAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0N
DQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8f
HyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDEx
MTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkND
Q0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVV
VVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdn
Z2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5
eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouL
i4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2d
nZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+v
r7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHB
wcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT
09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl
5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf3
9/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///yH5BAEKAP8ALAAAAAAWABYAAAjPAP8JHEiwoMGD
CBMqXMiw4cErWBwWLPPK1RWJAr+4etRIlReJWVR54oOn0qgsDa+AUjXoz547nDJdVHjFUq1F
hgT5wUOHVKOZDxP5mtRIEaJBeO7oWQUIaME9xDpZoiTpUSA/ffgEijXnIBxkzMJqyqSIkFlf
vXbVSlPwSpW3WZyBqpRo0SJFwbS8reKUYJVophw9iiQpErEqDKtI8/SI0iVMlowhXliFWqZI
ljZ5ynRsssLKkyBVyqTpUufE04YNM3asdTHPCffK3ouxdu2AADs=}

image create photo ::ttk::dialog::image::gohome -data {
R0lGODlhFgAWAMYAAKQAAPR1deU5OeInJ+xGRvFMTPBRUfJVVeAmJvNbW/JeXntMSohaWN4m
JvNkZJldW4SFgpubmsCKitwmJvRsbPRmZp11c4+Qjbi5uMLCwbq6ucShodwlJfNjY6ONi5+g
nr+/vt7e3d3d3dfX18m1tZwICKefnaOjotra2urq6unp6efn59zQ0IQiIaGgnqKjodjY2Obm
5uTk5OPj4+Le3tvc21VXU3d4enZ5fXV1dXV2dvPz8+7u7n6Ae3+BfICCfeXl5XZ5fHmZw3eY
wnV4fPLy8u3t7YSGgYWHguLi4nV4e1+Gt0p2rnJ1evHx8ezs7IaIg4qIZYmIcODg4HF4gTRl
pG52gfDw8Ovr64eJhIiJfvn5+bGztri8wbq7vLm9waSkpO/v74iKhd7e3qKioqOjo2VnY5eZ
lJiZlpmalpmal/j4+P//////////////////////////////////////////////////////
/////////////////////////yH5BAEKAH8ALAAAAAAWABYAAAf+gH+Cg4QAAISIiYUBAYeK
j38AjIyOkIsCjAONloOSBAWZmpWPkgYHjZIIopCSCQqNCwySDauJkg4OjQ8QERKSE7WdARQV
jRYXGBkaG5IcwZEBHY0eHyAhISIjJCUBHJvCjSYnKCnlKiorLNzfgpItLi8wMSv0MTIyMzTc
o5E1Nv//0N3AkQOHjh38/tjgYaOHjx8/YgAJIiTHECJFbCSyYcTGkY9IZCRRsiQHkyZONCKy
8cQGFChRpCSZQqVKjipWrqgkZAOLjSxZtPiYsoVLFy9fwITZOchGChtioooZs4VMmatlRDAV
ZOOKmTNo0qjZwaOs2TVbFQJcyxYgp7cEcDkFAgA7}

image create photo ::ttk::dialog::image::gohomebw -data {
R0lGODlhFgAWAOcAAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0N
DQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8f
HyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDEx
MTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkND
Q0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVV
VVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdn
Z2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5
eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouL
i4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2d
nZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+v
r7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHB
wcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT
09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl
5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf3
9/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///yH5BAEKAP8ALAAAAAAWABYAAAj+AP8JHEgwRgyC
CBMW3LTpoMKH/2IwZOgQ4kI2DL80tDhQ4p0+GTVWfCgREKGGEruIhCgRkaKGWc6kXJlQoiNH
Dd0Q0qRJIheaHTdRgtQQ0CNcwXKtkrgFaMRNOGNM+uSrm9Vru2hs2rIxaMNQorSpG5su3blp
WrsKlPgDlChs5s7JNUeO3LhvWkdG3Falb1+zd/DUETxP778q7qr4+QMIkLlyeCjVkXRHXpWE
VdpVIcS5EDlxd/7UcUMn3mWEVdhVMWSIUCFx4Ox0qdOFDrzTBKusq3Ko9x9w+WTt0sWL1Dvc
A6uoq4KoOSJv+USNmj6qG3KBVeCVuYQpU6Z57sIPi8d3/bDf8+j9clzPnmNAADs=}

image create photo ::ttk::dialog::image::reload -data {
R0lGODlhFgAWAOcAADtqqDtrqDdnpTVmpThopjpqpzdopjpqqHeaxaC726zG4q7I46jC3p25
2X6hyk16sDZnpTdnpjRlpFqDt7fN5bDI4qC82q3G4bfN5rrP5rvR57vQ57vR6K/H4W+VwThp
pnOYxUZ0rkVyrJ+52Ux4sDlppjdmpU56sYWmzbbN5bXM5abC4LHK5KO/3X+iy8PW6kNxrDtq
p1N+sz5tq0BvqzZnpDpppzprqH+kzLHJ45a325G02bTM5cja7EBuqjtrpzVlpE56tGKNw0x4
r6fA3a/J43Sfz83d7j1sqD9uq2yWyjpqqT5tqbTJ4pS22nKfz9Xi8DdopabA3cna7M3d7dTi
8Nzn8zZnpjRlpTlppzhopzZmpTVlpdrl8eDp8+Dp9OHq9Nnk8FV+szVmpOLr9aC+3qG+3dvm
8n+gx0FwrOPs9Zu63L3S6Nzm8lB8smWOxEd2sDxsqDRmpOTs9crb7K7I4sHV6oKjyzdopz5u
qz1sqUd0ruXt9sDS5tjj8dHf7qrG4sDU6bjN5YSkzGqQv1B7sj1rqEBuqVeBtZOy1sXV6Dlo
pjtppoelysjY6bDJ5LPK5KrE4Zq42j9vqURxqjxpo2qOvJSx07LJ4rbN5qK+3XKXwy9ZjzNl
pDFbkTZlojZno0FuqDdnpDZmpDRfl///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////yH5BAEKAP8ALAAAAAAWABYAAAj+AP8JHEiwoMGD
CBMWBBBAwAACBQwYHGCwwAEECRQsYNDAwQMIAyNIKFhgAoUKFi5gyKBhA4cOHj5EABGioIgR
JCKUMAHhBIoUKlawaOHiBQyCMWTEmEGjxkAbN3Dk0LGDRw8fBH8ACSLEhsEPQ4gUMXIECUEb
SZT8W3KQSRMnT6B4HShAYRQpU6hUsXJFIUEsEm5k0WLgChaCA7YoXszF78ABXbx8ARNGjMIx
BIGQKWPmDBqJBW8ITAMAsZo1bNq4yWLQwBs4EuIQlDOHTh07Vu7gASlwQB49MPYUlMOnj58/
gAIJGkSokKFDiFwkIlAQgqJFjBo5osBDxSMWkBYmRJI0yeAYSgMrWbqEiU2mHJo2leBksJNB
T59AhRI1ipTj/wD6FRAAOw==}

image create photo ::ttk::dialog::image::folder_new -data {
R0lGODlhEAAQAMYAAG1va2dpZc7OzsrEuW1pXvyxPsDAv5aXlZuYkOOrVb6cZODCkfvSkNyy
bMuYSfywPsnJyaamptm5hv3nw/765/740f3urPzJZvuyQGF6mjRlpDtnoFJwlNvFnv766P77
5P32u/3xkfzkddSsW2R4jMbY677S6MjMy+64Y/zTk/740/32vP3zo/zue/zoYPq+SNCgVK7H
44yx2I+w05yxwebFif3vrv3xkvzufPzrYfzfUe2+YV9hXdCyfvzLavzld/zoYfzgUfvDRNu6
gVVXU5OwzeGwYs2yf+a7Zvq9SeW6YNCxeem1ZT5onpWwy5Gw0J2xwOOxX7PF2ERrm4+x0p6x
vomu1qbC4Dlnoq3H44uw14qv14iu1oWr1YCo03qk0nOgz26dzpi53Fh2m5u73XCeznCdz26c
zm2czmybzmubzWqazmmZzWiZzZS226mrqYmv12uazWqazWiYzWaYzGWYzWWXzGSWzGOVzJq6
3lNxlkVdeT5giT9ghv///////yH5BAEKAH8ALAAAAAAQABAAAAfNgH8Ag4MBAX+IiYgAAo2O
AwSIBYoABgeXlwgJCgsMDQ4PghARpKUSExQVFhcYfwEGGRqyGxwdHh8gISIjJAEQGiUmJico
KSorLC0uLzCvGjEyMjM0NTY3ODk6Oxw8v9DRMj0+P0BBQkMaRAbP4EVGR0hJSktMTUTe4E5P
MlBRNDJSpqhjBy4alSozrFxJBwFLFi0ytGzh0sXLFzBhxKQzMIZMGTNhzqBJo2YNmzZu0r2B
oyaOHDZz6NSxcwdPHj17+MjayZNnH0VAgyIKBAA7}

image create photo ::ttk::dialog::image::configure -data {
R0lGODlhFgAWAMYAAH9/f+rp6Pn5+NjY1/j39vPy8YKXsjRlpOzq6P///050pHx8fLu7u+3r
6szMzK+vr2WErOjn5Orq6q2trePj4vr6+Xl4dNzc3JmZmejn5vLx7+3r6evp5/r6+oeHh8/N
yvj49/n59/n49/7+/p6enrW1tfb29Z+fnvj4+Ofm5LvByEVxqfT09J2wyt/f31l9q6enpiBK
h1R8rqSttvv7+3WQrqS60JCmvqexvcHBwePi4dPf6qKuvJ22zmN7lYScttfi7Y2YpZ240mB3
kZmtw9/n8IGTqVhthFtxiYaGhqG0yO7z9mB2j+Tj4V9fXpGmvvD095eltgAAALDG2+3y9oGK
lWyFocDR4v//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////yH5BAEKAH8ALAAAAAAWABYAAAfygH+CggAAg4eIiX8AAQID
hoqRhAQFj5EGB5EACAKQiAcJCpl/C4WCDA2diaAODxCEERIAExQIFRarFw+jfxgZGhsIHMOq
gwcdB7yCHh8gAiECwyKQByPKhySFACUCwiYnByjXkgACKSorLOOSiy0HLi8w7Icx9QcsMjM0
npIxNTY3YhzAkUPHCH6J/O3g0cNHDAAjhh2MFOMHkCA+hAyJsWiEsImIYhApYuSIkBpIOHaU
mISekiVGmJxMeQhiEycgYzyBEkUmSpWCpAgdKvRPjClUqswEGpToUKNWhFx5QjORUymDYlix
4lAS0ZD15hUVFAgAOw==}

image create photo ::ttk::dialog::image::folder -data {
R0lGODlhEAAQAKUAAG1va2dpZc7OzsbGxWNlYcDAv5aXlVVXU8nJyaampqenp6ioqGF6mjRl
pEZtnMbY677S6K7H44yx2F9hXYmu1qbC4Dlnoq3H44uw14qv14iu1oWr1YCo03qk0nOgz26d
zpi53Fh2m5u73XCeznCdz26czm2czmybzmubzWqazmmZzWiZzZS226mrqYmv12uazWqazWiY
zWaYzGWYzWWXzGSWzGOVzJq63lNxlkVdeT5giT9ghv///////////////yH5BAEKAD8ALAAA
AAAQABAAAAaGwB9gOAwEfsgkEiBoOgcEZRJQMFivhoNWu0QkvmCwYnFABgqMhnrNVjsCiMYD
Qq/bH41zIyLp+/8RDRNxfH+GgQcFe4aHDQeEjICOioWRFBWOCBYXGBIYGRobHB0eHyCTISIj
JB8lJicoKSorLI4tLigvMCoxMjM0NTY3ODk6bcdqO1LLy0EAOw==}

image create photo ::ttk::dialog::image::file -data {
R0lGODlhEAAQAIQAAJmZmYGBgf///+zs7Orq6uvr6+3t7fDw8MTExMXFxcbGxsfHx+7u7u3t
5e3t5u/v78jIyPHx8fLy8pWVlf//////////////////////////////////////////////
/yH5BAEKAB8ALAAAAAAQABAAAAVuIBCMZDl+aCCsbLsGqTAQRGEPg3EI8KcSiERCQVQsdr3f
DWcwMJCxwrBIPPKiBaahMXhefYIClcFweJOynJPxaEPBg+JiAam/VTmyO8L/qgxGdHV8En4C
TWwPBwcREoVoLpE9EyaVARMomZqbmiEAOw==}

### ttk::getOpenFile, ttk::getSaveFile, ttk::getAppendFile

proc ttk::getOpenFile {args} {
	return [::ttk::dialog::file::tkFDialog open $args]
}

proc ttk::getSaveFile {args} {
	return [::ttk::dialog::file::tkFDialog save $args]
}

proc ttk::getAppendFile {args} {
	return [::ttk::dialog::file::tkFDialog append $args]
}

proc ::ttk::dialog::file::getStyleColor {style option state} {
	if {[info vars ::tile::version] ne ""} {
		set stylecmd ::style
	} elseif {[info vars ::ttk::version] ne "" && 
		[info command ::ttk::style] ne ""} {
			set stylecmd ::ttk::style
	} else {
		return ""
	}
	return [$stylecmd lookup $style $option $state]
}

proc ::ttk::dialog::file::Create {win class} {
	toplevel $win -class $class
	wm withdraw $win

	set dataName [winfo name $win]
	upvar ::ttk::dialog::file::$dataName data

	set dirImg  ::ttk::dialog::image::folder
	set fileImg ::ttk::dialog::image::file

	# Additional frame to make sure the toplevel has the correct
	# background color for the theme
	#
	set w [ttk::frame $win.f]
	pack $w -fill both -expand 1

	# f1: the toolbar
	#
	set f1 [ttk::frame $w.f1 -class Toolbar]
	set data(bgLabel) [ttk::label $f1.bg -style Toolbutton]
	set data(upBtn) [ttk::button $f1.up -style Toolbutton]
	$data(upBtn) configure -image {::ttk::dialog::image::up 
		disabled ::ttk::dialog::image::upbw} \
		-command [list ::ttk::dialog::file::UpDirCmd $win]
	set data(prevBtn) [ttk::button $f1.prev -style Toolbutton]
	$data(prevBtn) configure -image {::ttk::dialog::image::previous
		disabled ::ttk::dialog::image::previousbw} \
		-command [list ::ttk::dialog::file::PrevDirCmd $win]
	set data(nextBtn) [ttk::button $f1.next -style Toolbutton]
	$data(nextBtn) configure -image {::ttk::dialog::image::next
		disabled ::ttk::dialog::image::nextbw} \
		-command [list ::ttk::dialog::file::NextDirCmd $win]
	set data(homeBtn) [ttk::button $f1.home -style Toolbutton]
	$data(homeBtn) configure -image {::ttk::dialog::image::gohome \
		disabled ::ttk::dialog::image::gohomebw} \
		-command [list ::ttk::dialog::file::HomeDirCmd $win]
	set data(reloadBtn) [ttk::button $f1.reload -style Toolbutton]
	$data(reloadBtn) configure -image ::ttk::dialog::image::reload \
		-command [list ::ttk::dialog::file::Update $win]
	set data(newBtn) [ttk::button $f1.new -style Toolbutton]
	$data(newBtn) configure -image ::ttk::dialog::image::folder_new \
		-command [list ::ttk::dialog::file::NewDirCmd $win]
	set data(cfgBtn) [ttk::menubutton $f1.cfg -style Toolbutton]
	set data(cfgMenu) [menu $data(cfgBtn).menu -tearoff 0]
	$data(cfgBtn) configure -image ::ttk::dialog::image::configure \
		-menu $data(cfgMenu)
	set data(dirMenuBtn) [ttk::combobox $f1.menu]
	$data(dirMenuBtn) configure \
		-textvariable ::ttk::dialog::file::${dataName}(selectPath)
	proc [namespace current]::selectPathChanged {n1 n2 op} [subst {
		$data(dirMenuBtn) selection clear
	}]
	trace add variable ::ttk::dialog::file::${dataName}(selectPath) write \
		[namespace current]::selectPathChanged

	set data(sortMenu) [menu $data(cfgMenu).sort -tearoff 0]
	set image [option get $data(cfgMenu) image Image]
	set selimage [option get $data(cfgMenu) selectImage Image]

	$data(cfgMenu) add cascade -label " Sorting" -menu $data(sortMenu) \
		-image $image -compound left
	$data(cfgMenu) add separator
	$data(cfgMenu) add radiobutton -label "Short View" -compound left \
		-image $image -selectimage ::ttk::dialog::image::radio16 \
		-variable ::ttk::dialog::file::details -value 0 -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-details ::ttk::dialog::file::details]
	$data(cfgMenu) add radiobutton -label "Detailed View" -compound left \
		-image $image -selectimage ::ttk::dialog::image::radio16 \
		-variable ::ttk::dialog::file::details -value 1 -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-details ::ttk::dialog::file::details]
	$data(cfgMenu) add separator
	$data(cfgMenu) add checkbutton -label "Show Hidden Files" \
		-image $image -selectimage $selimage -compound left \
		-variable ::ttk::dialog::file::hidden -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-hidden ::ttk::dialog::file::hidden]
	$data(cfgMenu) add checkbutton -label "Separate Folders" \
		-image $image -selectimage $selimage -compound left \
		-variable ::ttk::dialog::file::sepfolders -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-sepfolders ::ttk::dialog::file::sepfolders]
	$data(sortMenu) add radiobutton -label "By Name" -compound left \
		-image $image -selectimage ::ttk::dialog::image::radio16 \
		-variable ::ttk::dialog::file::sort -value name -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-sort ::ttk::dialog::file::sort]
	$data(sortMenu) add radiobutton -label "By Date" -compound left \
		-image $image -selectimage ::ttk::dialog::image::radio16 \
		-variable ::ttk::dialog::file::sort -value date -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-sort ::ttk::dialog::file::sort]
	$data(sortMenu) add radiobutton -label "By Size" -compound left \
		-image $image -selectimage ::ttk::dialog::image::radio16 \
		-variable ::ttk::dialog::file::sort -value size -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-sort ::ttk::dialog::file::sort]
	$data(sortMenu) add separator
	$data(sortMenu) add checkbutton -label "Reverse" \
		-image $image -selectimage $selimage -compound left \
		-variable ::ttk::dialog::file::reverse -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-reverse ::ttk::dialog::file::reverse]
	$data(sortMenu) add checkbutton -label "Folders First" \
		-image $image -selectimage $selimage -compound left \
		-variable ::ttk::dialog::file::foldersfirst -indicatoron 0 \
		-command [list ::ttk::dialog::file::setopt $win \
			-foldersfirst ::ttk::dialog::file::foldersfirst]

	$data(prevBtn) state disabled
	$data(nextBtn) state disabled
	if {![info exists ::env(HOME)]} {
		$data(homeBtn) state disabled
	}

	place $data(bgLabel) -relheight 1 -relwidth 1

	pack $data(upBtn) -side left -fill y
	pack $data(prevBtn) -side left -fill y
	pack $data(nextBtn) -side left -fill y
	pack $data(homeBtn) -side left -fill y
	pack $data(reloadBtn) -side left -fill y
	pack $data(newBtn) -side left -fill y
	pack $data(cfgBtn) -side left -fill y
	pack $data(dirMenuBtn) -side left -fill x -expand 1 -padx 8

	# f2: the frame with the OK button, cancel button, "file name" field,
	#     and file types field.
	#
	set f2 [ttk::frame $w.f2]
	ttk::label $f2.lab1 -text "Location:" -anchor w
	set data(location) [ttk::combobox $f2.loc]
	$data(location) configure \
		-textvariable ::ttk::dialog::file::${dataName}(selectFile)
	proc [namespace current]::locationChanged {n1 n2 op} [subst {
		$data(location) selection clear
	}]
	trace add variable ::ttk::dialog::file::${dataName}(selectFile) write \
		[namespace current]::locationChanged
	set data(typeMenuLab) [ttk::label $f2.lab2 -text "Filter:" -anchor w]
	set data(typeMenuBtn) [ttk::combobox $f2.filter -state readonly]
	set data(okBtn) [ttk::button $f2.ok -text OK -default active \
		-style Slim.TButton \
		-command [list ::ttk::dialog::file::Done $win]]
	set data(cancelBtn) [ttk::button $f2.cancel -text Cancel \
		-style Slim.TButton \
		-command [list ::ttk::dialog::file::Cancel $win]]

	grid $f2.lab1 $f2.loc $data(okBtn) -padx 4 -pady {5 0} -sticky ew
	grid $f2.lab2 $f2.filter $data(cancelBtn) -padx 4 -pady {5 0} -sticky ew
	grid columnconfigure $f2 1 -weight 1

	# f3: The file and directory lists
	#
    if {[string match "0.7.*" [package present tile]]} {
	set f3 [ttk::paned $w.f3 -orient horizontal]
    } else {
	set f3 [ttk::panedwindow $w.f3 -orient horizontal]
    }
	array set fontinfo [font actual [[label $f3.dummy] cget -font]]
	set font [list $fontinfo(-family) -14]
	destroy $f3.dummy
	$f3 add [ttk::frame $f3.dir] -weight 0
	ttk::label $f3.dir.bg -relief sunken
	set data(dirArea) [treectrl $f3.dir.t -bg white -width 160 -height 16 \
		-font $font -bd 1 -relief flat -highlightthickness 0 -cursor "" \
		-showroot no -showbuttons no -showlines no \
		-showheader no -takefocus 1 \
		-yscrollcommand [list $f3.dir.y set] \
		-xscrollcommand [list $f3.dir.x set]]
	ttk::scrollbar $f3.dir.y -command [list $f3.dir.t yview]
	ttk::scrollbar $f3.dir.x -command [list $f3.dir.t xview] \
		-orient horizontal
	$data(dirArea) column create -tag DIR
	$data(dirArea) element create eImg image -image $dirImg
	$data(dirArea) element create eTxt text -fill [list white {selected focus}] -lines 1
	$data(dirArea) element create eSel rect -fill [list #0a5f89 {selected focus} gray {selected !focus}]
	set S [$data(dirArea) style create STYLE]
	$data(dirArea) style elements $S {eSel eImg eTxt}
	$data(dirArea) style layout $S eImg -expand ns
	$data(dirArea) style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$data(dirArea) style layout $S eSel -union [list eTxt] -iexpand ns -ipadx 2 -ipady 1 -padx {1 4}
	configureTreectrlItemwidthequal $data(dirArea) yes
	grid $f3.dir.t $f3.dir.y -sticky ns
	grid $f3.dir.x -sticky we
	grid $f3.dir.bg -row 0 -column 0 -rowspan 2 -columnspan 2 -sticky news
	grid $f3.dir.t -sticky news -padx {2 0} -pady {2 0}
	grid columnconfigure $f3.dir 0 -weight 1
	grid rowconfigure $f3.dir 0 -weight 1

	$f3 add [ttk::frame $f3.file] -weight 1

	# The short view version
	#
	set data(short) [ttk::frame $f3.file.short]
	ttk::label $data(short).bg -relief sunken
	set data(fileArea) [treectrl $data(short).t -bg white -width 42 -height 16 \
		-font $font -bd 1 -relief flat -highlightthickness 0 -cursor "" \
		-showroot no -showbuttons no -showlines no \
		-orient vertical -wrap window \
		-showheader no -takefocus 1 \
		-xscrollcommand [list $data(short).x set]]
	ttk::scrollbar $data(short).x -orient horizontal \
		-command [list $data(short).t xview]
	$data(fileArea) column create -tag NAME
	$data(fileArea) state define directory
	$data(fileArea) state define file
	$data(fileArea) state define link
	$data(fileArea) element create eImg image -image [list $dirImg {directory} $fileImg {}]
	$data(fileArea) element create eTxt text -fill [list white {selected focus}] -lines 1
	$data(fileArea) element create eSel rect -fill [list #0a5f89 {selected focus} gray {selected !focus}]
	set S [$data(fileArea) style create STYLE]
	$data(fileArea) style elements $S {eSel eImg eTxt}
	$data(fileArea) style layout $S eImg -expand ns
	$data(fileArea) style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$data(fileArea) style layout $S eSel -union [list eTxt] -iexpand ns -ipadx 2 -ipady 1 -padx {1 4}
	configureTreectrlItemwidthequal $data(fileArea) no
	grid $data(short).t -sticky news -padx 2 -pady {2 0}
	grid $data(short).x -sticky ew
	grid $data(short).bg -row 0 -column 0 \
		-rowspan 2 -columnspan 2 -sticky news
	grid columnconfigure $data(short) 0 -weight 1
	grid rowconfigure $data(short) 0 -weight 1

	# The detailed view version
	#
	set data(long) [ttk::frame $f3.file.long]
	ttk::label $data(long).bg -relief sunken
	ttk::frame $data(long).f
	set data(fileList) [treectrl $data(long).t -width 320 -height 160 \
		-bg white -font $font -bd 0 -highlightthickness 0 -cursor "" -relief flat \
		-showroot no -showbuttons no -showlines no \
		-orient vertical -wrap "" -usetheme 1\
		-showheader yes -takefocus 1 \
		-yscrollcommand [list $data(long).y set] \
		-xscrollcommand [list $data(long).x set]]
	ttk::scrollbar $data(long).y -command [list $data(long).t yview]
	ttk::scrollbar $data(long).x -command [list $data(long).t xview] \
		-orient horizontal
	$data(fileList) state define directory
	$data(fileList) state define file
	$data(fileList) state define link
	$data(fileList) column create -tag NAME -text Name -width 140 -justify left
	$data(fileList) column create -tag SIZE -text Size -width 75  -justify right
	$data(fileList) column create -tag DATE -text Date -width 110 -justify left
	$data(fileList) column create -tag PERMISSIONS -text Permissions -width 60 -justify left
	$data(fileList) column create -tag OWNER -text Owner -justify left
	$data(fileList) column create -tag GROUP -text Group -justify left
	$data(fileList) column configure all -border 1 \
		-font TkHeadingFont -textpady 3 \
		-background [list \
			[getStyleColor TButton -background active] active \
			[getStyleColor TButton -background pressed] pressed \
			[getStyleColor TButton -background ""] {}]
	$data(fileList) element create eImg image -image [list $dirImg {directory} $fileImg {}]
	$data(fileList) element create eTxt text -fill [list white {selected focus}] -lines 1
	$data(fileList) element create eTxt2 text -lines 1
	$data(fileList) element create eSel rect -fill [list #0a5f89 {selected focus} gray {selected !focus}]
	set S [$data(fileList) style create STYLE_ICONTEXT]
	$data(fileList) style elements $S {eSel eImg eTxt}
	$data(fileList) style layout $S eImg -expand ns -padx {1 0}
	$data(fileList) style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$data(fileList) style layout $S eSel -union [list eTxt] -iexpand ns -ipadx 2 -ipady 1 -padx {1 4}
	set S [$data(fileList) style create STYLE_TEXT]
	$data(fileList) style elements $S {eSel eTxt2}
	$data(fileList) style layout $S eTxt2 -squeeze x -expand ns -iexpand ns -ipadx 2 -ipady 1 -padx {1 4} 
	configureTreectrlItemwidthequal $data(fileList) no

	grid $data(long).f $data(long).y -sticky ew -padx {2 0} -pady {1 0}
	grid $data(long).t ^ -sticky news -padx {2 0}
	grid $data(long).x -sticky ew
	grid $data(long).y -sticky ns -padx 0 -pady 0
	grid $data(long).bg -row 0 -column 0 \
		-rowspan 3 -columnspan 2 -sticky news
	grid columnconfigure $data(long) 0 -weight 1
	grid rowconfigure $data(long) 1 -weight 1

	grid $data(long) $data(short) -row 0 -column 0 -sticky news
	grid columnconfigure $f3.file 0 -weight 1
	grid rowconfigure $f3.file 0 -weight 1

	# Get rid of the default bindings
	bindtags $data(dirArea) [list $data(dirArea) FileDialogDir TreeCtrl $win all]
	bindtags $data(fileArea) [list $data(fileArea) FileDialogFile TreeCtrl $win all]
	bindtags $data(fileList) [list $data(fileList) FileDialogList TreeCtrl $win all]
	foreach T [list $data(fileArea) $data(fileList)] {
		$T notify bind $T <ActiveItem> {
			::ttk::dialog::file::FileSetName %T %c
		}
	}

	set data(paneWin) $f3

	pack $f1 -side top -fill x
	pack $f2 -side bottom -fill x -padx 8 -pady {5 5}
	pack $f3 -side bottom -fill both -expand 1 -padx 8 -pady {6 0}

	set data(history) ""
	set data(histpos) -1

	update idletasks
	pack propagate $w 0

	wm protocol $win WM_DELETE_WINDOW [list $data(cancelBtn) invoke]

	bind $data(dirMenuBtn) <Return> [list ::ttk::dialog::file::chdir $win]
	bind $data(dirMenuBtn) <<ComboboxSelected>> \
		[list ::ttk::dialog::file::chdir $win]
	bind $data(location) <Return> [list ::ttk::dialog::file::Done $win]
	bind $data(typeMenuBtn) <Return> \
		[list ::ttk::dialog::file::SetFilter $win]
	bind $data(typeMenuBtn) <<ComboboxSelected>> \
		[list ::ttk::dialog::file::SelectFilter $win]
}

proc ::ttk::dialog::file::ChangeDir {w dir} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	set data(history) [lrange $data(history) 0 $data(histpos)]
	set cwd [lindex $data(history) $data(histpos)]
	set data(selectPath) [file normalize [file join $cwd $dir]]
	lappend data(history) $data(selectPath)
	if {[incr data(histpos)]} {
		$data(prevBtn) state !disabled
		set data(selectFile) ""
	}
	$data(nextBtn) state disabled

	UpdateWhenIdle $w
}

proc ::ttk::dialog::file::UpdateWhenIdle {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	if {[info exists data(updateId)]} {
		return
	} elseif {[winfo ismapped $w]} {
		set after idle
	} else {
		set after 1
	}
	set data(updateId) [after $after [list ::ttk::dialog::file::Update $w]]
}

proc ::ttk::dialog::file::Update {w} {
	# This proc may be called within an idle handler. Make sure that the
	# window has not been destroyed before this proc is called
	if {![winfo exists $w]} {
		return
	}

	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data
	unset -nocomplain data(updateId)

	if {$data(-details)} {
		grid $data(long)
		grid remove $data(short)
	} else {
		grid $data(short)
		grid remove $data(long)
	}
	if {$data(-sepfolders)} {
		if {![llength [winfo manager $data(paneWin).dir]]} {
			$data(paneWin) insert 0 $data(paneWin).dir
		}
	} else {
		if {[llength [winfo manager $data(paneWin).dir]]} {
			$data(paneWin) forget 0
		}
	}

	$w configure -cursor watch
	update

	set dir ::ttk::dialog::image::folder
	set file ::ttk::dialog::image::file

	set cwd [lindex $data(history) $data(histpos)]

	if {$data(-hidden)} {
		set pattern "* .*"
	} else {
		set pattern "*"
	}

	# Make the directory list
	set dlist ""
	foreach f [eval glob -nocomplain -tails \
		-directory [list $cwd] -type d $pattern] {
		if {[string equal $f .]} continue
		if {[string equal $f ..]} continue
		lappend dlist [list $f dir]
	}

	# Make the file list	
	set flist ""
	set filter $data(filter)
	if {[string equal $filter *]} {
		set filter $pattern
	}
	foreach f [eval [linsert $filter 0 glob -nocomplain -tails \
		-directory $cwd -type {f l c b p}]] {
		# Links can still be directories. Skip those.
		if {[file isdirectory [file join $cwd $f]]} continue
		lappend flist [list $f file]
	}

	# Combine the two lists, if necessary
	if {$data(-sepfolders)} {
		set dlist [sort $w $dlist]
		set flist [sort $w $flist]
	} elseif {$data(-foldersfirst)} {
		set flist [concat [sort $w $dlist] [sort $w $flist]]
		set dlist ""
	} else {
		set flist [sort $w [concat $flist $dlist]]
		set dlist ""
	}
	
	set t $data(dirArea) 
	set selected_item [lindex [$t selection get] 0] 
	if {$selected_item ne ""} {
		set selected_text [$t item text $selected_item DIR]
	}
	$t item delete all
	foreach f $dlist {
		set item [$t item create]
		$t item collapse $item
		$t item style set $item DIR STYLE
		$t item text $item DIR [lindex $f 0]
		$t item lastchild root $item
		if {$selected_item ne "" && $selected_text eq [lindex $f 0]} {
			::TreeCtrl::SetActiveItem $t $item
			$t selection add $item
		}
	}
	if {[$t item numchildren root]} {
		$t see [$t item firstchild root]
	}

	set selected_item [lindex [$data(fileList) selection get] 0] 
	if {$selected_item ne ""} {
		set selected_text [$data(fileList) item text $selected_item NAME]
	} else {
		set selected_item [lindex [$data(fileArea) selection get] 0] 
		if {$selected_item ne ""} {
			set selected_text [$data(fileArea) item text $selected_item NAME]
		}
	}

	if {$data(-details)} {
		set t $data(fileList)
		$t item delete all
		set size ""
		set date ""
		set mode ""
		set uid ""
		set gid ""
		foreach f $flist {
			lassign $f name type size date mode uid gid
			if {![info exists users($uid)] || \
				![info exists groups($gid)]} {
				set fname [file join $cwd $name]
				# May fail for dead links
				if {![catch {array set attr \
					[file attributes $fname]}]} {
					if {[info exists attr(-owner)]} {
						set users($uid) $attr(-owner)
					} else {
						set users($uid) ""
					}
					if {[info exists attr(-group)]} {
						set groups($gid) $attr(-group)
					} else {
						set groups($gid) ""
					}
				}	
			}
			catch {set uid $users($uid)}
			catch {set gid $groups($gid)}

			set item [$t item create]
			$t item collapse $item
			$t item style set $item NAME STYLE_ICONTEXT
			$t item style set $item SIZE STYLE_TEXT
			$t item style set $item DATE STYLE_TEXT
			$t item style set $item PERMISSIONS STYLE_TEXT
			$t item style set $item OWNER STYLE_TEXT
			$t item style set $item GROUP STYLE_TEXT
			$t item text $item NAME $name
			$t item text $item SIZE [list [expr {$size / 1024}] KB]
			$t item text $item DATE [datefmt $date]
			$t item text $item PERMISSIONS [modefmt $type $mode]
			$t item text $item OWNER $uid
			$t item text $item GROUP $gid

			if {$type eq "file"} {
				$t item state set $item file
			} elseif {$type eq "directory"} {
				$t item state set $item directory
			} elseif {$type eq "link"} {
				$t item state set $item link
			}
			$t item lastchild root $item
			if {$selected_item ne "" && $selected_text eq $name} {
				::TreeCtrl::SetActiveItem $t $item
				$t selection add $item
			}
		}
	} else {
		set t $data(fileArea)
		$t item delete all
		foreach f $flist {
			lassign $f name type
			set item [$t item create]
			$t item collapse $item
			$t item style set $item NAME STYLE
			$t item text $item NAME $name
			if {$type eq "file"} {
				$t item state set $item file
			} elseif {$type eq "directory"} {
				$t item state set $item directory
			} elseif {$type eq "link"} {
				$t item state set $item link
			}
			$t item lastchild root $item
			if {$selected_item ne "" && $selected_text eq $name} {
				::TreeCtrl::SetActiveItem $t $item
				$t selection add $item
			}
		}
	}

	if {[$t item numchildren root]} {
		$t see [$t item firstchild root]
	}

	if {[string equal $cwd "/"]} {
		$data(upBtn) state disabled
	} else {
		$data(upBtn) state !disabled
	}
	$w configure -cursor ""
}

proc ::ttk::dialog::file::sort {w list} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	set cwd [lindex $data(history) $data(histpos)]
	set order [expr {$data(-reverse) ? "-decreasing" : "-increasing"}]
	set newlist ""
	foreach f $list {
		set file [lindex $f 0]
		# Use lstat in case the destination doesn't exists
		file lstat [file join $cwd $file] stat
		if {[string equal $stat(type) link]} {
			# This may fail if the link points to nothing
			if {![catch {file stat [file join $cwd $file] dest}]} {
				array set stat [array get dest]
				if {[string equal $stat(type) file]} {
					set stat(type) link
				}
			}
		}
		lappend newlist [list $file $stat(type) $stat(size) \
			$stat(mtime) $stat(mode) $stat(uid) $stat(gid)]
	}
	switch -- $data(-sort) {
		size {
			set mode -integer
			set idx 2
		}
		date {
			set mode -integer
			set idx 3
		}
		default {
			set mode -dictionary
			set idx 0
		}
	}
	lsort $order $mode -index $idx $newlist
}

proc ::ttk::dialog::file::datefmt {str} {
	clock format $str -format {%x %R}
}

proc ::ttk::dialog::file::modefmt {type mode} {
	switch $type {
		file {set rc -}
		default {set rc [string index $type 0]}
	}
	binary scan [binary format I $mode] B* bits
	foreach b [split [string range $bits end-8 end] ""] \
		c {r w x r w x r w x} {
		if {$b} {append rc $c} else {append rc -}
	}
	set rc
}

proc ::ttk::dialog::file::setopt {w option var} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data
	upvar #0 $var value

	
	set data($option) $value
	UpdateWhenIdle $w
}

proc ::ttk::dialog::file::UpDirCmd {w} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	ChangeDir $w [file dirname [lindex $data(history) $data(histpos)]]
}

proc ::ttk::dialog::file::PrevDirCmd {w} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	set data(selectFile) ""
	incr data(histpos) -1
	set data(selectPath) [lindex $data(history) $data(histpos)]
	$data(nextBtn) state !disabled
	if {!$data(histpos)} {
		$data(prevBtn) state disabled
	}
	Update $w
}

proc ::ttk::dialog::file::NextDirCmd {w} {
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	set data(selectFile) ""
	incr data(histpos)
	set data(selectPath) [lindex $data(history) $data(histpos)]
	$data(prevBtn) state !disabled
	if {$data(histpos) >= [llength $data(history)] - 1} {
		$data(nextBtn) state disabled
	}
	Update $w
}

proc ::ttk::dialog::file::HomeDirCmd {w} {
	ChangeDir $w ~
}

proc ::ttk::dialog::file::NewDirCmd {win} {
	set dataName [winfo name $win]
	upvar ::ttk::dialog::file::$dataName data

	set dir [lindex $data(history) $data(histpos)]

	toplevel $win.new
	wm title $win.new "New Folder"
	set w [ttk::frame $win.new.f]
	pack $w -expand 1 -fill both

	ttk::label $w.prompt -anchor w -justify left \
		-text "Create new folder in:\n$dir"
	ttk::entry $w.box -width 36 -validate all \
		-validatecommand [list ::ttk::dialog::file::NewDirVCmd $w %P]
	ttk::separator $w.sep
	set f [ttk::frame $w.buttons]
	ttk::button $f.clear -text Clear -takefocus 0 \
		-command [list $w.box delete 0 end]
	ttk::button $f.ok -text OK -default active \
		-command [list ::ttk::dialog::file::NewDirExit $win 1]
	ttk::button $f.cancel -text Cancel \
		-command [list ::ttk::dialog::file::NewDirExit $win]
	grid $f.clear $f.ok $f.cancel -padx 4 -pady {0 10} -sticky we
	grid columnconfigure $f {0 1 2} -uniform 1
	pack $w.prompt $w.box $w.sep $f \
		-side top -padx 12 -pady 3 -anchor w -fill x
	pack $w.prompt -pady {12 0}
	pack $f -anchor e -fill none -padx 8
	wm transient $win.new $win
	wm resizable $win.new 0 0
	wm protocol $win.new WM_DELETE_WINDOW [list $f.cancel invoke]

	bind $w.box <Return> [list $f.ok invoke]

	::tk::PlaceWindow $win.new widget $win
	::tk::SetFocusGrab $win.new $w.box
}

proc ::ttk::dialog::file::NewDirVCmd {w str} {
	if {[string length $str]} {
		$w.buttons.ok state !disabled
		$w.buttons.clear state !disabled
	} else {
		$w.buttons.ok state disabled
		$w.buttons.clear state disabled
	}
	return 1
}

proc ::ttk::dialog::file::NewDirExit {w {save 0}} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	if {$save} {
		set dir [lindex $data(history) $data(histpos)]
		set newdir [file join $dir [$w.new.f.box get]]
		if {[catch {file mkdir $newdir} err]} {
			ttk::messageBox -type ok -parent $w.new -icon error \
				-message "$err"
			return
		} else {
			ChangeDir $w $newdir
		}
	}
	destroy $w.new
	::tk::RestoreFocusGrab $w.new $w.new.f.box
}

proc ::ttk::dialog::file::Cancel {w} {
	variable selectFilePath ""
}

proc ::ttk::dialog::file::Done {w} {
	variable selectFilePath
	variable filelist
	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	if {![string length $data(selectFile)] || \
		[string equal $data(selectFile) .]} {
		return -code break
	}

	set cwd [lindex $data(history) $data(histpos)]
	set path [file join $cwd $data(selectFile)]

	if {[file isdirectory $path]} {
		ChangeDir $w $path
		return -code break
	}

	if {![string length [file extension $path]]} {
		append path $data(-defaultextension)
	}

	if {[file exists $path]} {
		if {[string equal $data(type) save]} {
			set reply [ttk::messageBox -icon warning -type yesno \
				-parent $w -message "File\
				\"$path\" already exists.\nDo\
				you want to overwrite it?"]
			if {[string equal $reply "no"]} {return}
		}
	} else {
		if {[string equal $data(type) open]} {
			ttk::messageBox -icon warning -type ok -parent $w \
				-message "File \"$path\" does not exist."
			return
		}
	}

	set idx [lsearch -exact $filelist $path]
	set filelist [linsert [lreplace $filelist $idx $idx] 0 $path]

	set selectFilePath $path
	return -code break
}

proc ::ttk::dialog::file::chdir {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	set dir $data(selectPath)
	if {[file isdirectory $dir]} {
		ChangeDir $w $dir
	} else {
		ttk::messageBox -type ok -parent $w \
			-message "Cannot change to the directory\
				\"$data(selectPath)\".\nPermission denied." \
			-icon warning
	}
	return -code break
}

proc ::ttk::dialog::file::SelectFilter {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	set data(filter) [lindex $data(-filetypes) \
		[$data(typeMenuBtn) current] 1]
	::ttk::dialog::file::UpdateWhenIdle $w
}

proc ::ttk::dialog::file::SetFilter {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	set data(filter) [$data(typeMenuBtn) get]
	::ttk::dialog::file::UpdateWhenIdle $w
	return -code break
}

proc ::ttk::dialog::file::IsArrow {key} {
	if {$key eq "Up"
	 || $key eq "Down"
	 || $key eq "Left"
	 || $key eq "Right"} {
		 return 1
	 }
	 return 0
}

proc ::ttk::dialog::file::ActivateFirstItem {w} {
	set first [$w item firstchild 0]
	if {$first ne ""} {
		::TreeCtrl::SetActiveItem $w $first
	}
	return 
 }

proc ::ttk::dialog::file::DirButton1 {w x y} {
	FileClearSelectedName $w
	focus $w
}

proc ::ttk::dialog::file::DirRelease1 {w x y} {
	set top [winfo toplevel $w]
	$top configure -cursor ""
}

proc ::ttk::dialog::file::DirDouble1 {w x y} {
	set id [$w identify $x $y]
	set item [lindex $id 1]
	if {$item ne ""} {
		set dir [$w item text $item DIR]
		ChangeDir [winfo toplevel $w] $dir
	}
}

proc ::ttk::dialog::file::DirMotion1 {w x y} {
	[winfo toplevel $w] configure -cursor "X_cursor #C00 #000"
}

proc ::ttk::dialog::file::DirKey {w key} {
	if {$key eq "BackSpace"} {
		::ttk::dialog::file::UpDirCmd [winfo toplevel $w]
		return "break"
	}

	set item [lindex [$w selection get] 0]
	if {$item eq ""} {
		if {[IsArrow $key]} {
			ActivateFirstItem $w
			return "break"
		}
		return
	}
	switch -exact -- $key {
		"Return" {
			set dir [$w item text $item DIR]
			ChangeDir [winfo toplevel $w] $dir
		}
		default {return} 
	}
}

proc ::ttk::dialog::file::FileButton1 {w x y} {
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data
	variable filetype

	lassign [$w identify $x $y] type item
	if {[info var item] eq ""} return
	if {$item eq ""} return
	if {$type eq "header"} {
		FileSort $w $item
		return
	}
	
	if {[$w item state get $item "file"]} {
		set filetype "file"
	} elseif {[$w item state get $item "directory"]} {
		set filetype "directory"
	} elseif {[$w item state get $item "link"]} {
		set filetype "link"
	} else {
		set filetype "none"
	}
	if {$filetype eq "none"} return
	FileClearSelectedName $w

	if {$filetype eq "file" || $filetype eq "link"} {
		set data(selectFile) [$w item text $item NAME]
	}
	focus $w
}

proc ::ttk::dialog::file::FileRelease1 {w x y} {
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data
	variable filetype

	set top [winfo toplevel $w]
	if {[llength [$top cget -cursor]]} {
		# The mouse has been moved, don't perform the action
		$top configure -cursor ""
	}
	[winfo toplevel $w] configure -cursor ""
	set filetype none
}

proc ::ttk::dialog::file::FileMotion1 {w x y} {
	[winfo toplevel $w] configure -cursor "X_cursor #C00 #000"
}

proc ::ttk::dialog::file::FileDouble1 {w x y} {
	lassign [$w identify $x $y] type item
	if {[info var item] eq ""} return
	if {$item eq ""} return
	if {$type eq "header"} return

	if {[$w item state get $item "directory"]} {
		set dir [$w item text $item NAME]
		ChangeDir [winfo toplevel $w] $dir
	} else {
		catch { ::ttk::dialog::file::Done [winfo toplevel $w] }
	}
}

proc ::ttk::dialog::file::FileKey {w key} {
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data

	if {$key eq "BackSpace"} {
		::ttk::dialog::file::UpDirCmd [winfo toplevel $w]
		return "break"
	}

	set item [lindex [$w selection get] 0]
	if {$item eq ""} {
		if {[IsArrow $key]} {
			ActivateFirstItem $w
			return "break"
		 }
		 return
	}
	
	switch -exact -- $key {
		"Up" {
			return [::ttk::dialog::file::FileKeyMoveUpDownExt $w "up" $item]
		}
		"Down" {
			return [::ttk::dialog::file::FileKeyMoveUpDownExt $w "down" $item]
		}
		"Return" {
			if {[$w item state get $item "directory"]} {
				set dir [$w item text $item NAME]
				ChangeDir [winfo toplevel $w] $dir
			} else {
				catch { ::ttk::dialog::file::Done [winfo toplevel $w] }
			}
		}
		default {return} 
	}
}

proc ::ttk::dialog::file::FileClearSelectedName {w} {
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data
	$data(fileArea) selection clear
	$data(fileList) selection clear
	$data(dirArea) selection clear
}

proc ::ttk::dialog::file::FileKeyMoveUpDownExt {w direct prev} {
	set rnc [$w item rnc $prev]
	if {$rnc eq ""} {
		return
	}
	lassign $rnc row column
	set rows [$w item id {1 bottom}]
	if {$direct eq "up" && $row == 0} {
		set next [$w item prevsibling $prev]
	} elseif {$direct eq "down" && (($row+1) % $rows) == 0} {
		set next [$w item nextsibling $prev]
	} elseif {$direct eq "down" && 
		[$w item compare $prev == [$w item id last]]} {
		set next $prev
	} else {
		return
	}
	if {$next eq ""} {
		return
	}
	FileClearSelectedName $w
	::TreeCtrl::SetActiveItem $w $next
	$w selection add $next
	return "break"
}

proc ::ttk::dialog::file::FileSetName {w item} {
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data
	
	if {[$w item state get $item "directory"]} {
		return
	}
	set data(selectFile) [$w item text $item NAME]
}

proc ::ttk::dialog::file::FileSort {w col} {
	variable sort
	variable reverse
	set dataName [winfo name [winfo toplevel $w]]
	upvar ::ttk::dialog::file::$dataName data
	set next_sort [lindex {name size date} [$w column index $col]]
	if {$next_sort eq ""} {
		return
	}
	if {$next_sort eq $sort} {
		set reverse [expr {$reverse ? 0:1}]
		setopt [winfo toplevel $w] -reverse ::ttk::dialog::file::reverse
	} else {
		set sort $next_sort
		setopt [winfo toplevel $w] -sort ::ttk::dialog::file::sort
	}
}

proc ::ttk::dialog::file::tkFDialog {type arglist} {
	global env
	variable selectFilePath
	variable filelist
	set dataName __ttk_filedialog
	upvar ::ttk::dialog::file::$dataName data

	::ttk::dialog::file::Config $dataName $type $arglist

	if {[string equal $data(-parent) .]} {
		set w .$dataName
	} else {
		set w $data(-parent).$dataName
	}

	if {![winfo exists $w]} {
		::ttk::dialog::file::Create $w TkFDialog
	} elseif {![string equal [winfo class $w] TkFDialog]} {
		destroy $w
		::ttk::dialog::file::Create $w TkFDialog
	} else {
		$data(fileArea) item delete all
		$data(dirArea) item delete all
		$data(prevBtn) state disabled
		$data(nextBtn) state disabled
		$data(upBtn) state disabled
		set data(history) ""
		set data(histpos) -1
	}

	wm transient $w $data(-parent)

	if {[llength $data(-filetypes)]} {
		set titles ""
		foreach type $data(-filetypes) {
			lassign $type title filter
			lappend titles $title
		}
		$data(typeMenuBtn) configure -values $titles
		$data(typeMenuLab) state !disabled
		$data(typeMenuBtn) state !disabled
		$data(typeMenuBtn) current 0
		::ttk::dialog::file::SelectFilter $w
	} else {
		set data(filter) "*"
		$data(typeMenuBtn) configure -takefocus 0
		$data(typeMenuBtn) state disabled
		$data(typeMenuLab) state disabled
	}

	set dirlist "/"
	if {[info exists env(HOME)] && ![string equal $env(HOME) /]} {
		lappend dirlist $env(HOME)
	}
	if {[lsearch -exact $dirlist $data(selectPath)] < 0} {
		lappend dirlist $data(selectPath)
	}
	foreach n $filelist {
		set dir [file dirname $n]
		if {[lsearch -exact $dirlist $dir] < 0} {
			lappend dirlist $dir
		}
	}
	$data(dirMenuBtn) configure -values $dirlist
	$data(location) configure -values $filelist

	::ttk::dialog::file::ChangeDir $w $data(selectPath)

	::tk::PlaceWindow $w widget $data(-parent)
	wm title $w $data(-title)

	::tk::SetFocusGrab $w $data(location)

	tkwait variable ::ttk::dialog::file::selectFilePath

	::tk::RestoreFocusGrab $w $data(location) withdraw

	return $selectFilePath
}

proc ::ttk::dialog::file::Config {dataName type argList} {
	upvar ::ttk::dialog::file::$dataName data

	set data(type) $type

	# 1: the configuration specs
	#
	set specs {
		{-defaultextension "" "" ""}
		{-filetypes "" "" ""}
		{-initialdir "" "" ""}
		{-initialfile "" "" ""}
		{-parent "" "" "."}
		{-title "" "" ""}
		{-sepfolders "" "" 1}
		{-foldersfirst "" "" 1}
		{-sort "" "" "name"}
		{-reverse "" "" 0}
		{-details "" "" 0}
		{-hidden "" "" 0}
	}

	# 2: default values depending on the type of the dialog
	#
	if {![info exists data(selectPath)]} {
		# first time the dialog has been popped up
		set data(selectPath) [pwd]
		set data(selectFile) ""
	}

	# 3: parse the arguments
	#
	tclParseConfigSpec ::ttk::dialog::file::$dataName $specs "" $argList

	if {$data(-title) == ""} {
		if {[string equal $type "save"]} {
			set data(-title) "Save As"
		} else {
			set data(-title) "Open"
		}
	}

	# 4: set the default directory and selection according to the -initial
	#    settings
	#

	# Ensure that initialdir is an absolute path name.
	if {[string length $data(-initialdir)]} {
		set dir [file normalize [file join [pwd] $data(-initialdir)]]
		if {[string equal [file type $dir] "link"]} {
			set dir [file normalize [file join $dir [file link $dir]]]
		}
		if {[file isdirectory $dir]} {
			set data(selectPath) $dir
		} else {
			set data(selectPath) [pwd]
		}
	}
	set data(selectFile) $data(-initialfile)

	# 5. Parse the -filetypes option
	#
	set data(-filetypes) [::tk::FDGetFileTypes $data(-filetypes)]

	if {![winfo exists $data(-parent)]} {
		error "bad window path name \"$data(-parent)\""
	}

	variable sepfolders $data(-sepfolders)
	variable foldersfirst $data(-foldersfirst)
	variable sort $data(-sort)
	variable reverse $data(-reverse)
	variable details $data(-details)
	variable hidden $data(-hidden)
}

### ttk::chooseDirectory

proc ::ttk::dialog::file::treeCreate {w} {
	destroy $w
	toplevel $w -class TkChooseDir
	wm iconname $w Dialog

	set dataName [winfo name $w]
	upvar ::ttk::dialog::file::$dataName data

	if {[winfo viewable [winfo toplevel $data(-parent)]] } {
		wm transient $w $data(-parent)
	}

	set f1 [ttk::frame $w.f1]
	set data(dirMenuBtn) [ttk::combobox $f1.dir \
		-textvariable ::ttk::dialog::file::${dataName}(selectPath)]
	pack $f1.dir -fill x -expand 1 -padx 8 -pady 5

	set f2 [ttk::frame $w.f2]
	ttk::frame $f2.f
	ttk::label $f2.f.bg -relief sunken
	array set fontinfo [font actual [[label $f2.f.dummy] cget -font]]
	set font [list $fontinfo(-family) -14]
	destroy $f2.f.dummy
	set data(dirTree) [treectrl $f2.f.tree -width 300 -height 200 -font $font \
		-highlightthickness 0 -bd 1 -cursor "" -bg white -relief flat \
		-showroot no -showbuttons yes -showlines yes -showheader no -takefocus 1 \
		-yscrollcommand [list $f2.f.scroll set]]
	ttk::scrollbar $f2.f.scroll -command [list $data(dirTree) yview]
	$data(dirTree) state define checked
	$data(dirTree) column create -tag DIR
	$data(dirTree) element create eImg image -image ::ttk::dialog::image::folder
	$data(dirTree) element create eTxt text -fill [list white {selected focus}] -lines 1
	$data(dirTree) element create eSel rect -fill [list #0a5f89 {selected focus} gray {selected !focus}]
	set S [$data(dirTree) style create STYLE]
	$data(dirTree) style elements $S {eSel eImg eTxt}
	$data(dirTree) style layout $S eImg -expand ns
	$data(dirTree) style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$data(dirTree) style layout $S eSel -union [list eTxt] -iexpand ns -ipadx 2 -ipady 1 -padx {1 4}
	$data(dirTree) configure -treecolumn DIR
	grid $f2.f.tree $f2.f.scroll -sticky ns
	grid $f2.f.tree -sticky news -padx {2 0} -pady {2 2}
	grid $f2.f.bg -column 0 -row 0 -columnspan 2 -rowspan 2 -sticky news
	grid columnconfigure $f2.f 0 -weight 1
	grid rowconfigure $f2.f 0 -weight 1
	pack $f2.f -fill both -expand 1 -padx 8 -pady 4

	set f3 [ttk::frame $w.f3]
	ttk::button $f3.ok -text OK -default active \
		-command [list ::ttk::dialog::file::TreeDone $w]
	ttk::button $f3.cancel -text Cancel \
		-command [list ::ttk::dialog::file::Cancel $w]
	grid x $f3.ok $f3.cancel -sticky ew -padx {4 8} -pady 8
	grid columnconfigure $f3 {1 2} -uniform buttons -minsize 80
	grid columnconfigure $f3 0 -weight 1

	pack $f1 -side top -fill x
	pack $f3 -side bottom -fill x
	pack $f2 -side top -fill both -expand 1

	# Get rid of the default Text bindings
	bindtags $data(dirTree) [list $data(dirTree) DirDialog TreeCtrl $w all]
	$data(dirTree) notify bind $data(dirTree) <ActiveItem> {
		::ttk::dialog::file::treeSetName [winfo toplevel %T] %c
	}
	$data(dirTree) notify bind $data(dirTree) <Expand-after> {
		::ttk::dialog::file::treeOpen [winfo toplevel %T] %I
		update
	}
	$data(dirTree) notify bind $data(dirTree) <Collapse-after> {
		update
	}

	bind $data(dirMenuBtn) <Return> \
		[list ::ttk::dialog::file::TreeReturn $w]

	wm protocol $w WM_DELETE_WINDOW [list $f3.cancel invoke]
}

# return a avsolute path 
proc ::ttk::dialog::file::treeGetPath {tree item} {
	set path [$tree item text $item DIR]
	while {[set item [$tree item parent $item]]} {
		set dir [$tree item text $item DIR]
		set path [file join $dir $path]
	}
	return $path
}

# return a item that have $dirname in the $parent
proc ::ttk::dialog::file::treeGetItem {tree parent dirname} {
	set parent [$tree item id $parent]
	if {$parent ne ""} {
		foreach item [$tree item children $parent] {
			if {[$tree item text $item DIR] eq $dirname} {
				return $item
			}
		}
	}
	return ""
}

proc ::ttk::dialog::file::treeUpdate {w dir} {
	upvar ::ttk::dialog::file::[winfo name $w](dirTree) tree

	$tree item delete all
	foreach v [file volumes] {
		set v [string toupper $v]
		set item [$tree item create]
		$tree item collapse $item
		$tree item configure $item -button yes
		$tree item style set $item DIR STYLE
		$tree item text $item DIR $v
		$tree item lastchild root $item
	}
	treeRecursiveOpen $w $dir
}

proc ::ttk::dialog::file::treeRecursiveOpen {w path} {
	upvar ::ttk::dialog::file::[winfo name $w](dirTree) tree
	set dir [file normalize $path]
	set list [file split $dir]
	lappend list .

	set parent root
	foreach d $list {
		treeOpen $w $parent active $d
		set parent [treeGetItem $tree $parent $d]
		if {$parent eq ""} break
	}
	TreeSelect $w active
	$tree see [$tree item id last]
	$tree see active
}

proc ::ttk::dialog::file::treeOpen {w parent {index insert} {subdir .}} {
	upvar ::ttk::dialog::file::[winfo name $w](dirTree) tree

	if {[$tree item id $parent] == 0} {
		return
	}

	set path [treeGetPath $tree $parent]

	# This may take a little longer so give some indication to the user
	$w configure -cursor watch
	update

	if {![$tree item state get $parent checked]} {
		set list [glob -nocomplain -tails -dir $path -type d * .*]
		if {[llength $list] > 2} {
			foreach d [lsort -dictionary $list] {
				# Skip . and ..
				if {[string equal $d .] || [string equal $d ..]} continue

				set dirPath [file normalize [file join $path $d]]

				file stat $dirPath stat
				if {$stat(nlink) != 2} {
					set hasButton 1
				} else {
					set hasButton 0
				}

				set item [$tree item create]
				$tree item collapse $item
				$tree item configure $item -button $hasButton
				$tree item style set $item DIR STYLE
				$tree item text $item DIR $d
				$tree item lastchild $parent $item

				# Put a mark if this is the specified subdirectory
				if {[string equal $d $subdir]} {
					::TreeCtrl::SetActiveItem $tree $item
				}
			}
		} else {
			# parent doesn't have children. 
			# remove open/close indicator button.
			$tree item configure $parent -button no
		}
		$tree item state set $parent checked
	} else {
		set item [treeGetItem $tree $parent $subdir]
		if {$item ne ""} {
			::TreeCtrl::SetActiveItem $tree $item
		}
	}
	
	if {![$tree item state get $parent open]} {
		$tree item expand $parent
	}
	$w configure -cursor ""
}

proc ::ttk::dialog::file::treeClose {w index} {
	upvar ::ttk::dialog::file::[winfo name $w](dirTree) tree
	set item [$tree item id $index]
	if {$item ne ""} {
		$tree item collapse $item
	}
}

proc ::ttk::dialog::file::TreeDone {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	if {[file exists $data(selectPath)]} {
		if {![file isdirectory $data(selectPath)]} {
			return
		}
	} elseif {[string is true $data(-mustexist)]} {
		return
	}
	variable selectFilePath $data(selectPath)
}

proc ::ttk::dialog::file::cdTree {w dir {subdir .}} {
	treeRecursiveOpen $w [file normalize [file join $dir $subdir]]
}

proc ::ttk::dialog::file::treeSetName {w item} {
	upvar ::ttk::dialog::file::[winfo name $w] data
	set tree $data(dirTree)

	set data(selectPath) [treeGetPath $tree $item]
}

proc ::ttk::dialog::file::TreeSelect {w index} {
	upvar ::ttk::dialog::file::[winfo name [winfo toplevel $w]] data
	set tree $data(dirTree)
	set item [$tree item id $index]
	if {$item ne ""} {
		$tree selection clear
		$tree selection add $item
		set data(selectPath) [treeGetPath $tree $item]
	}
}

proc ::ttk::dialog::file::TreeReturn {w} {
	upvar ::ttk::dialog::file::[winfo name $w] data

	if {[file isdirectory $data(selectPath)]} {
		set tree $data(dirTree)
		cdTree $w $data(selectPath)
		TreeSelect $w active
		$tree see [$tree item id last]
		$tree see active
	}

	return -code break
}

proc ::ttk::dialog::file::TreeKey {tree key} {
	set item [lindex [$tree selection get] 0]
	if {$item eq ""} {
		return
	}
	set w [winfo toplevel $tree]

	switch -exact -- $key {
		"Left" {
			if {![$tree item state get $item open] ||
				![$tree item cget $item -button]} {
				set parent [$tree item parent $item]
				if {[$tree compare 0 != $parent]} {
					::TreeCtrl::SetActiveItem $tree $parent
					TreeSelect $w active
					$tree see $item
				}
			} else {
				treeClose $w $item
			}
		}
		"Right" {
			if {![$tree item state get $item open]} {
				treeOpen $w $item
			} else {
				set child [$tree item firstchild $item]
				if {$child ne ""} {
					::TreeCtrl::SetActiveItem $tree $child
					TreeSelect $w active
					$tree see $child
				}
			}
		}
		"Return" {
		}
	}
}

proc ttk::dialog::file::configureTreectrlItemwidthequal {w val} {
	if {[string match "2.0*" [package present treectrl]]} {
		$w column configure all -widthhack $val
	} else {
		$w configure -itemwidthequal $val
	}
}

proc ttk::chooseDirectory {args} {
	set dataName __ttk_dirdialog
	upvar ::ttk::dialog::file::$dataName data

	set specs {
		{-initialdir "" "" .}
		{-mustexist "" "" 0}
		{-parent "" "" .}
		{-title "" "" ""}
    	}
	tclParseConfigSpec ::ttk::dialog::file::$dataName $specs "" $args

	if {$data(-title) == ""} {
		set data(-title) "[::tk::mc "Choose Directory"]"
	}

	if {![winfo exists $data(-parent)]} {
		error "bad window path name \"$data(-parent)\""
	}

	if {[string equal $data(-parent) .]} {
		set w .$dataName
	} else {
		set w $data(-parent).$dataName
	}

	if {![winfo exists $w]} {
		::ttk::dialog::file::treeCreate $w
	}

	::tk::PlaceWindow $w widget $data(-parent)
	wm title $w $data(-title)
	::tk::SetFocusGrab $w $data(dirTree)

	::ttk::dialog::file::treeUpdate $w $data(-initialdir)

	tkwait variable ::ttk::dialog::file::selectFilePath

	::tk::RestoreFocusGrab $w $data(dirTree) withdraw

	return $::ttk::dialog::file::selectFilePath
}

# Alternative procedure names
interp alias {} ttk_getOpenFile {} ::ttk::dialog::file::tkFDialog open
interp alias {} ttk_getSaveFile {} ::ttk::dialog::file::tkFDialog save
interp alias {} ttk_getAppendFile {} ::ttk::dialog::file::tkFDialog append


# Need to have a lassign procedure
if {![llength [info procs lassign]]} {
	proc lassign {list args} {
		uplevel 1 [list foreach $args $list break]
		lrange $list [llength $args] end
	}
}
if {![llength [info commands treectrl]]} {
	package require treectrl 2.1
}


set styleconfig configure
if {[info vars ::tile::version] ne ""} {
    foreach {v1 v2 v3} [split $::tile::version .] {break}
    if {($v2 * 100 + $v3) < 702} {
        set styleconfig default
    }
}
style $styleconfig Slim.TButton -padding 0
option add *TkFDialog*selectBackground #0a5f89
option add *TkFDialog*selectForeground #ffffff
option add *TkFDialog*Toolbar*takeFocus 0
option add *TkFDialog*Text.background white
option add *TkFDialog*Menu.activeBackground #0a5f89
option add *TkFDialog*Menu.activeForeground #ffffff
option add *TkFDialog*Menu.activeBorderWidth 1
option add *TkFDialog*Menu.borderWidth 1
option add *TkFDialog*Menu.relief solid
option add *TkFDialog*Menu.Image ::ttk::dialog::image::blank16
option add *TkFDialog*Menu*selectImage ::ttk::dialog::image::tick16

# Bindings
bind FileDialogDir <ButtonPress-1> {::ttk::dialog::file::DirButton1 %W %x %y}
bind FileDialogDir <ButtonRelease-1> {::ttk::dialog::file::DirRelease1 %W %x %y}
bind FileDialogDir <Double-1> {::ttk::dialog::file::DirDouble1 %W %x %y}
bind FileDialogDir <B1-Motion> {::ttk::dialog::file::DirMotion1 %W %x %y}
bind FileDialogDir <Key> {
	if {[::ttk::dialog::file::DirKey %W %K] eq "break"} {
		break
	}
}
bind FileDialogFile <ButtonPress-1> {::ttk::dialog::file::FileButton1 %W %x %y}
bind FileDialogFile <ButtonRelease-1> {::ttk::dialog::file::FileRelease1 %W %x %y}
bind FileDialogFile <B1-Motion> {::ttk::dialog::file::FileMotion1 %W %x %y}
bind FileDialogFile <Double-1> {::ttk::dialog::file::FileDouble1 %W %x %y}
bind FileDialogFile <Key> {
	if {[::ttk::dialog::file::FileKey %W %K] eq "break"} {
		break
	}
}
bind FileDialogList <ButtonPress-1> {::ttk::dialog::file::FileButton1 %W %x %y}
bind FileDialogList <ButtonRelease-1> {::ttk::dialog::file::FileRelease1 %W %x %y}
bind FileDialogList <B1-Motion> {::ttk::dialog::file::FileMotion1 %W %x %y}
bind FileDialogList <Double-1> {::ttk::dialog::file::FileDouble1 %W %x %y}
bind FileDialogList <Key> {::ttk::dialog::file::FileKey %W %K}
bind DirDialog <Key> {::ttk::dialog::file::TreeKey %W %K}
bind DirDialog <Double-1> {
	::ttk::dialog::file::treeOpen [winfo toplevel %W] [lindex [%W selection get] 0]
}

interp alias {} ::tk_getOpenFile     {} ::ttk::getOpenFile 
interp alias {} ::tk_getSaveFile     {} ::ttk::getSaveFile 
interp alias {} ::tk_chooseDirectory {} ::ttk::chooseDirectory
}

# Database info
array set database {
	path     {}
	name     {} 
	encoding utf-8
	version  3
}

#
# DebugTool
#
namespace eval Debug {;#<<<
namespace export assert dputs deval
}
proc Debug::enable {flg} {
	if {!$flg} {
		proc ::Debug::assert args {}
		proc ::Debug::dputs args {}
		proc ::Debug::deval args {}
		return
	}
	# assert {expr}
	# ex. assert {$n == 1}
	proc ::Debug::assert args {
		uplevel 1 " if $args {return -code ok}
		return -code error \{assertion error : $args\} "
	}

	# dputs {argString}
	# ex. dputs {key=$key, value=$val}
	proc ::Debug::dputs args {
		if {[catch {uplevel 1 eval "puts \"$args\""} r]} {
			puts $r
		}
	}

	# deval {script}
	# ex. deval {puts $n}
	proc ::Debug::deval args {
		uplevel 1 eval $args
	}
}
Debug::enable 0
;#>>>

Debug::enable $::pref(debug)
namespace import -force Debug::dputs
namespace import -force Debug::assert
namespace import -force Debug::deval
dputs {*** Debug Mode ***}

#
# Utilities 
#
namespace eval Util {;#<<<

	# Is Number?
	proc isNumber {str} {
		if {$str eq {}} {return 0}
		if {[string is integer $str]} {return 1}
		if {[string is double $str]} {return 1}
		return 0
	}
	
	# lintersect {a b c d e} {c d e f g} returns {c d e}
	proc lintersect {a b} {
		if {[llength $a]==0 || [llength $b]==0} {return [list]}
		if {[llength $a] > [llength $b]} {
			set L $a
			set S $b
		} else {
			set L $b
			set S $a
		}
		set S [lsort -uniq $S]
		set ret ""
		foreach e $S {
			set n [lsearch -exact $L $e]
			if {$n >= 0} {
				set L [lreplace $L $n $n]
				lappend ret $e
			}
		}
		return $ret
	}

	# lexcept {a b c d e} {c d e f g} returns {a b}
	proc lexcept {a b} {
		set ret ""
		foreach e $a {
			set n [lsearch -exact $b $e]
			if {$n >= 0} {
				lappend ret $e
			}
		}
		return $ret
	}

	# increment x,y format. incrIndex 0,0 , 1 0
	proc incrIndex {index sep {xi 0} {yi 0}} {
		set l [split $index $sep]
		foreach {x y} $l {
			if {$x eq {}} {set x 0}
			if {$y eq {}} {set y 0}
			incr x $xi
			incr y $yi
			return $x$sep$y
		}
	}
	
	# parse sequens number
	# {1 2 3 4 7 8 9} is converted to {{1 4} {7 3}}.
	proc exactConsecutiveNumber {inlist} {
		if {[llength $inlist] == 0} return
		set nlist [lsort -integer -uniq $inlist]
		set ret [list]
		
		set seqS [lindex $nlist 0]
		set i $seqS
		foreach n [lrange $nlist 1 end] {
			incr i
			if {$n != $i} {
				lappend ret [list $seqS [expr {$i - $seqS}]]
				set seqS $n
				set i    $n
			}
		}
		incr i
		lappend ret [list $seqS [expr {$i - $seqS}]]
		return $ret
	}
	
	# Python's range like
	proc range {from to {step 1}} {
		set res $from
		while {$to>$from} {lappend res [incr from $step]}
		return $res
	}
	
	# Compare dict types. if equal return 0, else 1.
	# nest is compare level. if 0, compare with only first level.
	# if nest is -1, compare deep.
	proc dictCompare {a b {nest -1}} {
		if {[dict size $a] != [dict size $b]} {
			return 1
		}
		set nnest [expr {$nest < 0 ? -1 : $nest - 1}]
		dict for {key val} $a {
			if {[dict exists $b $key] == 0} {return 1}
			set bval [dict get $b $key]
			if {($nest != 0) && ([llength $val] % 2 == 0)} {
				if {[dictCompare $bval $val $nnest]} {
					return 1
				}
			} elseif {$bval ne $val} {
				return 1
			}
		}
		return 0
	}
	
	# Select random
	proc randselect {vdata} {
		return [lindex $vdata [expr {int(rand() * [llength $vdata])}]]
	}

	# make pop and return it.
	# $root.pop   PopMenu path (created here)
	# $root.text  Text Widget path
	proc makeEditorPop {root} {
		set shortmod [getModifierKey short]
		set modifier [getModifierKey]
		set m $root.pop
		if {![winfo exists $m]} { menu $m }
		$m add command -label [msgcat::mc Cut] -accelerator "$shortmod+X" \
			-command "tk_textCut $root.text"
		$m add command -label [msgcat::mc Copy] -accelerator "$shortmod+C" \
			-command "tk_textCopy $root.text"
		$m add command -label [msgcat::mc Paste] -accelerator "$shortmod+V" \
			-command "tk_textPaste  $root.text"
		$m add command -label [msgcat::mc Delete] -accelerator "$shortmod+D" \
		-command "event generate $root.text <$modifier-d>"
		$m add separator
		$m add command -label [msgcat::mc "Select All"] -accelerator "$shortmod+A" \
			-command "event generate $root.text <$modifier-a>"
		return $m
	}
	
	# bind to common pop.
	proc bindEditorPopCommon {w} {
		set shortmod [getModifierKey short]
		set modifier [getModifierKey]
		set m ._editorpop
		if {![winfo exists $m]} {
			menu $m -tearoff 0
			$m add command -label [msgcat::mc Cut] -accelerator "$shortmod+X" \
			-command {event generate [focus] <<Cut>>}
			$m add command -label [msgcat::mc Copy] -accelerator "$shortmod+C" \
			-command {event generate [focus] <<Copy>>}
			$m add command -label [msgcat::mc Paste] -accelerator "$shortmod+V" \
			-command {event generate [focus] <<Paste>>}
			$m add command -label [msgcat::mc Delete] -accelerator "$shortmod+D" \
			-command {event generate [focus] <$modifier-d>}
			$m add separator
			$m add command -label [msgcat::mc "Select All"] -accelerator "$shortmod+A" \
			-command {event generate [focus] <$modifier-a>}
		}
		bind $w <<B3>> "if {\[$w cget -state\] ne \"disabled\"} {
					focus $w; tk_popup $m %X %Y}; break;"
	}
	
	proc quoteValue {val nullString} {
		if {$val eq $nullString} {
			return "NULL"
		}
		if {[isNumber $val]} {
			return $val
		}
		return "'[string map {' ''} $val]'"
	}
	
	proc unquoteValue {val} {
		if {[string index $val 0] eq "'" &&
			[string index $val end] eq "'"} {
			return [string range $val 1 end-1]
		} else {
			return $val
		}
	}
    
    proc getDefaultValue {sqliteversion type value} {
        set valuelen [string length $value]

        if {[string index $value 0] eq "(" && [string index $value [expr {$valuelen - 1}]] eq ")"} {
            return $value
        }

        switch -exact [string tolower $type] {
            date -
            time -
            datetime -
            timestamp {
                if {[versionCompare $sqliteversion "3.1.0"] >= 0} {
                    set uvalue [string toupper $value]
                    if {$uvalue eq "CURRENT_TIME" || $uvalue eq "CURRENT_DATE" || $uvalue eq "CURRENT_TIMESTAMP"} {
                        return $value
                    }
                }
            }
            int -
            smallint -
            float -
            real -
            money -
            decimal -
            number -
            integer {
                if {[isNumber $value]} {
                    return $value
                }
            }
        }
        return "'[string map {' ''} $value]'"
    }

	# set correct width and height of withdrawing window
	proc updateWindowSize {root {size {}}} {
		if {[tk windowingsystem] eq "x11"} {
			if {[wm state $root] eq "withdrawn"} {
				update
				set w [winfo reqwidth  $root]
				set h [winfo reqheight $root]
				wm geometry $root ${w}x${h}
			}
			if {$size ne ""} {wm geometry $root $size}
			update
		}
		if {[tk windowingsystem] eq "win32"} {
			if {[wm state $root] eq "withdrawn"} {
				set x [winfo x $root]
				set y [winfo y $root]
				wm geometry $root $size+10000+10000
				wm deiconify $root
				update idletask
				wm withdraw $root
				wm geometry $root +$x+$y
			} else {
				if {$size ne ""} {wm geometry $root $size}
				update idletask
			}
		}
	}
	
	proc moveCenter {widget {size {}}} {
		set sw [winfo vrootwidth $widget]
		set sh [winfo vrootheight $widget]
		if {[llength $size] == 0} {
			update idletask
			set w [winfo width $widget]
			set h [winfo height $widget]
		} else {
			set w [lindex $size 0]
			set h [lindex $size 1]
		}
		wm geometry $widget +[expr {($sw-$w)/2}]+[expr {($sh-$h)/2}]
	}
	
	# Nested grab command
	# ngrab set .w   : stack window and grab it.
	# ngrab release  : release a top of stack window.
	variable grabStack [list]
	proc ngrab {command {w {}}} {
		variable grabStack
		switch -exact -- $command {
			set {
				if {$w eq {}} { error "wrong args : ngrab command window" }
				grab set $w
				focus -force $w
				lappend grabStack $w
			}
			release {
				set cur [lindex $grabStack end]
				set grabStack [lrange $grabStack 0 "end-1"]
				if {[llength $grabStack] == 0} {
					grab release $cur
					return
				}
				grab set [lindex $grabStack end]
			}
		}
	}
	
	# bind mouse event to button widgets
	proc bindMouseEvent {w} {
		bind $w <KeyPress-Return> {
			event generate %W <ButtonPress-1>
		}
		bind $w <KeyRelease-Return> {
			event generate %W <ButtonRelease-1>
		}
	}
	
	# set validate command of ttk::entry. allow user to enter one char.
	proc setOneCharValidateToTtkentry {w} {
		$w configure -validate all \
			-validatecommand {
				if {[string length %S] > 0} {
					%W delete 0 end
					%W insert 0 [string index %S 0]
					return 1
				}
				return 0
			}
	}

	# get image format name
	proc getImageFormat {dataName} {
		upvar $dataName data
		# jpeg
		binary scan $data H6 hex
		if {$hex eq "ffd8ff"} {return jpeg}
		# gif
		binary scan $data a6 head
		if {$head eq "GIF87a" || $head eq "GIF89a"} {
			return gif
		}
		# png
		binary scan $data H16 hex
		if {$hex eq "89504e470d0a1a0a"} {return png}
		# tiff
		binary scan $data H2H2s1 b0 b1 s1
		if {$b0 eq $b1 && ($b1 eq "49" || $b1 eq "4d") && $s1 == 42} {
			return tiff
		}
		# tga
		binary scan $data cccsscsssscc\
		numid maptyp imgtyp maporig mapsize mapbits \
		xorig yorig xsize ysize pixsize imgdes
		set numid   [expr {$numid   & 0xff}]
		set maptyp  [expr {$maptyp  & 0xff}]
		set imgtyp  [expr {$imgtyp  & 0xff}]
		set mapbits [expr {$mapbits & 0xff}]
		set pixsize [expr {$pixsize & 0xff}]
		set imgdes  [expr {$imgdes  & 0xff}]
		if {($imgtyp == 2 || $imgtyp == 10) && 
			($pixsize == 24 || $pixsize == 32) &&
			!($xsize < 1 || $ysize < 1)} {
			if {$mapsize > 0} {
				if {[lsearch {15 16 24 32} $mapbits] > -1} {
					return tga
				}
			} else {return tga}
		}
		# sun
		binary scan $data H8 hex
		if {$hex eq "59a66a95"} {return sun}
		# sgi
		binary scan $data H4 hex
		if {$hex eq "01da"} {return sgi}
		# pcx
		binary scan $data ccccssssssc48ccssc58 \
		manufacture version compression bpp \
		x1 y1 x2 y2 hdpi vdpi \
		colormap reserved planes \
		bytesperline color filter
		set manufacture [expr {$manufacture & 0xff}]
		set version     [expr {$version     & 0xff}]
		set compression [expr {$compression & 0xff}]
		set bpp         [expr {$bpp         & 0xff}]
		set reserved    [expr {$reserved    & 0xff}]
		set planes      [expr {$planes      & 0xff}]
		if {$manufacture == 10 && ($bpp == 1 || $bpp == 8) &&
			($planes == 1 || $planes == 3 || $planes == 4)} {
			return pcx
		}
		# bmp
		binary scan $data a2ccccccccccccccccccccccccc head \
		b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 \
		b15 b16 b17 b18 b19 b20 b21 b22 b23 b24
		for {set i 0} {$i <= 24} {incr i} {
			set b$i [expr {[set b$i] & 0xff}]
		}
		if {$head eq "BM" && $b13 == 0 && $b14 == 0 && $b15 == 0} {
			switch -- $b12 {
				40 -
				64 {
				set widthPtr   [expr {($b19<<24) + ($b18<<16) + ($b17<<8) + $b16}]
				set heightPtr  [expr {($b23<<24) + ($b22<<16) + ($b21<<8) + $b20}]
				}
				12 {
				set widthPtr   [expr {($b17<<8) + $b16}]
				set heightPtr  [expr {($b19<<8) + $b18}]
				}
				default {
				set widthPtr   0
				set heightPtr  0
				}
			}
			if {$widthPtr > 0 && $heightPtr > 0} {
				return bmp
			}
		}
		# ico
		binary scan $data s1s1s1 reserved type nicons
		set reserved [expr {$reserved & 0xffff}]
		set type     [expr {$type     & 0xffff}]
		set nicons   [expr {$nicons   & 0xffff}]
		if {$reserved == 0 && $type == 1 && $nicons > 0} {
			return ico
		}
		# xpm
		# xbm
		# ppm
		binary scan $data a3 type
		if {$type eq "P6\n" || $type eq "P5\n"} {
			return ppm
		}
		# unknown
		return ""
	}

	# read csv and return tcl list
	proc skipChar {strName posName {char " "}} {
		upvar $strName str
		upvar $posName pos
		while {[string index $str $pos] eq $char} {
			incr pos
		}
		return $pos
	}

	proc readCSVToken {strName posName fp sep quote isLE} {
		upvar $strName str
		upvar $posName pos
		
		if {[string index $str $pos] ne $quote} {
			return [readCSVTokenWithoutQuote str pos $fp $sep $quote $isLE]
		}
		set st $pos
		set ed -1
		
		while {1} {
			set ed [string first $quote $str [expr {$pos + 1}]]
			if {$ed != -1} {
				set pos [expr {$ed + 1}]
				if {[string index $str $pos] ne $quote} {
					break
				}
			} else {
				if {[eof $fp]} {break}
				append str \n[Unicode::readLine $fp $isLE]
			}
		}
		
		if {$ed == -1} {
			set pos -1
			return [string range $str [incr st] end]
		}
		set ret [string range $str [incr st] [incr ed -1]]
		if {[string index $str $pos] ne $sep} {
			append ret [readCSVTokenWithoutQuote str pos $fp $sep $quote $isLE]
		}
		return $ret
	}

	proc readCSVTokenWithoutQuote {strName posName fp sep quote isLE} {
		upvar $strName str
		upvar $posName pos
		set st $pos
		set ed [string first $sep $str $st]

		set pos $ed
		
		if {$ed == -1} {
			return [string range $str $st end]
		} else {
			return [string trimright [string range $str $st [incr ed -1]] " "]
		}
	}

	# read csv and return tcl list
	proc readCSVLine {fp sep quote isLE {lenmax 4000}} {
		if {[eof $fp]} return

		set str [string trimleft [Unicode::readLine $fp $isLE] " "]

		set len [string length $str]
		set map [list $quote$quote $quote]
		if {$len == 0} return
		set data [list]
		set pos 0
		set ed 0
		while {1} {
			set stchar [string index $str $pos]
			if {$stchar eq $sep} {lappend data {}; incr pos; continue}

			set token [readCSVToken str pos $fp $sep $quote $isLE]
			lappend data [string map $map $token]
			
			if {$pos < 0 || $pos >= [string length $str]} {break}
			incr pos
			while {[string index $str $pos] eq " "} {
				incr pos
			}
		}
		return $data
	}
	
	# Unicode read/write
	namespace eval Unicode {
		proc _swapEndian {data} {
			binary scan $data s* str
			return [binary format S* $str]
		}
		
		proc _readUTF16Line {fp isLittleEndian} {
			if {$isLittleEndian} {
				set crlf \x0d\x00\x0a\x00
				set cr   \x0d\x00
				set lf   \x0a\x00
			} else {
				set crlf \x00\x0d\x00\x0a
				set cr   \x00\x0d
				set lf   \x00\x0a
			}

			set str ""
			set enc   [fconfigure $fp -encoding]
			set trans [lindex [fconfigure $fp -translation] 0]
			fconfigure $fp -translation binary -encoding binary
			switch -exact -- $trans {
				cr {
					while {! [eof $fp]} {
						set c [::read $fp 2]
						if {$c eq $cr} {
							break
						} else {
							append str $c
						}
					}
				}
				lf {
					while {! [eof $fp]} {
						set c [::read $fp 2]
						if {$c eq $lf} {
							break
						} else {
							append str $c
						}
					}
				}
				crlf {
					while {! [eof $fp]} {
						set c [::read $fp 2]
						if {$c eq $cr} {
							set c2 [::read $fp 2]
							if {$c2 eq $lf} {
								break
							} else {
								append str $c$c2
							}
						} else {
							append str $c
						}
					}
				}
				auto {
					while {! [eof $fp]} {
						set c [::read $fp 2]
						if {$c eq $lf} {
							break
						}
						if {$c eq $cr} {
							set c2 [::read $fp 2]
							if {$c2 eq $lf} {
								break
							} else {
								seek $fp -2 current
								break
							}
						} else {
							append str $c
						}
					}
				}
				binary -
				default {
					set str [::read $fp]
				}
			}

			fconfigure $fp -translation $trans -encoding $enc

			if {$::tcl_platform(byteOrder) eq "littleEndian"} {
				if {$isLittleEndian} {
					return [encoding convertfrom unicode $str]
				}
			} else {
				if {! $isLittleEndian} {
					return [encoding convertfrom uniocde $str]
				}
			}
			return [encoding convertfrom unicode [_swapEndian $str]]
		}

		proc readBOM {fp} {
			set enc [fconfigure $fp -encoding]
			set offset [tell $fp]
			if {$enc ne "utf-8" && $enc ne "unicode"} {
				return
			}
			fconfigure $fp -encoding binary
			
			set type ""

			if {$enc eq "utf-8"} {
				set n [binary scan [read $fp 3] H* bom]
				if {$n > 0} {
					if {$bom eq "efbbbf"} {
						set type "utf-8"
					} else {
						seek $fp $offset start
					}
				}
			}

			if {$enc eq "unicode"} {
				set n [binary scan [read $fp 2] H* bom]
				if {$n > 0} {
					switch -exact -- $bom {
						feff {
							set type "utf-16be"
						}
						fffe {
							set type "utf-16le"
						}
						default {
							seek $fp $offset start
						}
					}
				}
			}

			fconfigure $fp -encoding $enc
			return $type
		}
		
		proc writeUTF16BOM {fp} {
			puts -nonewline $fp \uFEFF
		}

		proc isLittleEndian {fp} {
			set bom [readBOM $fp]
			switch -exact -- $bom {
				utf-16le {set isLittleEndian 1}
				utf-16be {set isLittleEndian 0}
				default  {
					set isLittleEndian [expr {$::tcl_platform(byteOrder) eq "littleEndian"}]
				}
			}
			return $isLittleEndian
		}

		proc readAll {fp isLittleEndian} {
			set enc [fconfigure $fp -encoding]
			if {$enc ne "unicode"} {
				return [::read $fp]
			}
			
			if {$::tcl_platform(byteOrder) eq "littleEndian"} {
				if {$isLittleEndian} {
					return [::read $fp]
				}
			} else {
				if {! $isLittleEndian} {
					return [::read $fp]
				}
			}

			set lines ""
			while {! [eof $fp]} {
				append lines [readLine $fp $isLittleEndian]
				if {! [eof $fp]} {
				append lines \n
				}
			}
			return $lines
		}

		proc readLine {fp isLittleEndian} {
			set enc [fconfigure $fp -encoding]
			if {$enc ne "unicode"} {
				return [::gets $fp]
			}

			if {$::tcl_platform(byteOrder) eq "littleEndian"} {
				if {$isLittleEndian} {
					return [::gets $fp]
				}
			} else {
				if {! $isLittleEndian} {
					return [::gets $fp]
				}
			}
			return [_readUTF16Line $fp $isLittleEndian]
		}
	}

	# return 1(v1 > v2), -1(v1 < v2), 0(v1 eq v2)
	proc versionCompare {v1 v2} {
		set v1 [split $v1 .]
		set v2 [split $v2 .]
		set v1_len [llength $v1]
		set v2_len [llength $v2]
		if {$v1_len > $v2_len} {
			set max $v2_len
		} else {
			set max $v1_len
		}
		for {set i 0} {$i < $max} {incr i} {
			set vn1 [lindex $v1 $i]
			set vn2 [lindex $v2 $i]
			set vn1IsNum [string is digit $vn1]
			set vn2IsNum [string is digit $vn2]
			if {$vn1IsNum && !$vn2IsNum} {
				return 1
			}
			if {!$vn1IsNum && $vn2IsNum} {
				return -1
			}
			if {!$vn1IsNum && !$vn2IsNum} {
				if {$vn1 eq $vn2} {continue}
				set minv [lindex [lsort -dict [list $vn1 $vn2]] 0]
				if {$minv eq $vn1} {
					return -1
				} else {
					return 1
				}
			} else {
				if {$vn1 > $vn2} {return 1}
				if {$vn1 < $vn2} {return -1}
			}
		}
		if {$v1_len == $v2_len} {
			return 0
		}
		return [expr {$v1_len > $v2_len ? 1 : -1}]
	}

	# return modifier key depended on each platform (osx or anothers)
	proc getModifierKey {{isShort ""}} {
		if {[tk windowingsystem] eq "aqua"} {
			return Command
		}
		if {$isShort eq "short"} {
			return Ctrl
		}
		return Control
	}

	# right-click
	proc installVirtualEvent {} {
		if {[tk windowingsystem] eq "aqua"} {
			event add <<B3>> \
				<Control-ButtonPress-1> \
				<ButtonPress-2> \
				<ButtonPress-3>
			foreach class {Table Listbox Text TEntry TCombobox} {
				bind $class <2> {break}
				bind $class <3> {break}
				bind $class <B2-Motion> {break}
				bind $class <B3-Motion> {break}
				bind $class <Control-B1-Motion> {break}
				bind $class <Control-ButtonRelease-1> {break}
			}
			event delete <<Cut>> <Key-F2> 
			event delete <<Copy>> <Key-F3>
			event delete <<Paste>> <Key-F4>

			rename ::tk_popup ::core_tk_popup
			proc ::tk_popup {menu x y {entry {}}} {
				set classes {Table Listbox Text TEntry TCombobox}

				foreach class $classes {
					set binds [bind $class]
					set scripts ""
					foreach b $binds {
						lappend scripts [bind $class $b]
						bind $class $b {}
					}
					lappend class_scripts $scripts
					lappend class_binds   $binds
				}

				core_tk_popup $menu $x $y $entry

				foreach class $classes binds $class_binds scripts $class_scripts {
					foreach b $binds s $scripts {
						bind $class $b $s
					}
				}
			}

			bind Table <ButtonRelease-1> {
				if {$::tk::table::Priv(mouseMoved) &&
					$::tk::table::Priv(borderInfo) == "" &&
					[winfo exists %W]} {
				::tk::table::CancelRepeat
				%W activate @%x,%y
				}
			}
		} else {
			event add <<B3>> <ButtonRelease-3> 
		}
	}
	if {[info command tk] ne ""} {installVirtualEvent}

	namespace eval Color {}
	# args is color list. return colors median value
	proc Color::median {args} {
		set i 0
		set rsum 0
		set gsum 0
		set bsum 0
		foreach color $args {
			foreach {r g b} [winfo rgb . $color] {
				incr rsum $r
				incr gsum $g
				incr bsum $b
			}
			incr i
		}
		set i [expr {$i * 257}]
		return [format "#%02x%02x%02x" \
			[expr {$rsum / $i}]\
			[expr {$gsum / $i}]\
			[expr {$bsum / $i}]]
	}
}
;#>>>


#
# Tooltip Object
#
namespace eval Tooltip {;#<<<
	# How to use
	# Tooltip::init
	# Tooltip::bindWin .button "message"

	variable info
	array set info {}
	variable tooltip ._tooltip

	proc _show {win x y} {
		variable tooltip
		variable info
		wm geometry $tooltip +$x+[incr y 20]
		$tooltip.info config -text $info($win)
		wm deiconify $tooltip
		raise $tooltip
	}
	
	proc _hide {} {
		variable tooltip
		wm withdraw $tooltip
	}
	
	proc _addWin {win str} {
		variable info
		set info($win) $str
	}
	
	proc init {} {	
		variable tooltip
		option add *Balloonhelp*background white widgetDefault
		option add *Balloonhelp*foreground black widgetDefault
		
		toplevel $tooltip -class Balloonhelp \
			-background black -borderwidth 1 -relief flat
		wm withdraw $tooltip
		wm overrideredirect $tooltip 1
		
		label $tooltip.info -pady 0 -justify left
		pack $tooltip.info -side left -fill both
		return $tooltip
	}
	
	proc bindWin {win str} {
		_addWin $win $str
		bind $win <Enter> {+ ::Tooltip::_show %W %X %Y}
		bind $win <Leave> {+ ::Tooltip::_hide }
	}
}
;#>>>

Tooltip::init

#
# Ttk MessageBox
#
# This was a wrapper of ttk::dialog for using it like tk_messageBox.
# But the ttk::dialog was removed from Tk8.5a7 core recently.
# So I wrote missing code here. Now the Message::show doesn't required
# ttk::dialog. But the behavior is same. Note that icon images 
# (busy and auth) are lost.
namespace eval Message {;#<<<
	variable retval ""

	variable icon
	array set icon ""

	variable label
	array set label [list \
		ok     [msgcat::mc OK] \
		cancel [msgcat::mc Cancel] \
		yes    [msgcat::mc Yes] \
		no     [msgcat::mc No] \
		retry  [msgcat::mc Retry] \
	]

	variable type 
	array set type {
	ok {-icon info -buttons {ok} -default ok}
	okcancel {-icon info -buttons {ok cancel} -default ok -cancel cancel}
	yesno {-icon question -buttons {yes no}}
	yesnocancel {-icon question -buttons {yes no cancel} -cancel cancel}
	retrycancel {-icon question -buttons {retry cancel} -cancel cancel}
	}

	variable config
	set config(margin)     6   ;# space between icon and text
	set config(interspace) 6   ;# horizontal space between buttons
	set config(sepspace)   24  ;# vertical space above buttons
	set config(textwidth)  400 ;# width of dialog box text (pixels)
}

#
# load ttk
#
if {[info command ttk::button] eq ""} {
	package require tile
}

#
# Dialog icons
#
namespace eval Message {
set icon(info) [image create photo ::Message::icon::info -data {
R0lGODlhIAAgAPcAAAqwvc7k5nq8wU21vjeosrbQ0nrU3ErO2CmuuPb6+6jF
xzrI1HXM03TBx5G5vSq8x6TU2A+6x1y9xVjN15rV2db09xqtucTd4IvS2HzK
0Ee+xzq8xpbM0CfCzu/392XQ2FvHz3vGzSC5xSqzvhq2wYve5UTN19fp6nve
3p7e5Dm0vgq2xMjv80rG0VLO1o7Kz3XV3ZTQ1a3Fxb/T1Bi8yRi4xTXF0YPH
zWjGzmG3v2jP1zvAzJXEyI7Cx1rU3v///zSosQiywDK6xUS0vTK0vqPc4ELO
3oTc4lOwuNvq7GS9xCO/zITL0ZDZ36fHyh+tuVTDzDTL13Pa40rQ28Dm6eD2
+YXT2ZTc4vb4+HLIz5vQ1YC7wEPAysLU1ZnMzJ3c4avO0VXT3VG/x57m7GnV
3TvN2J7GySvAzTKttg+1whCxviK1wWPW3xm/zSOyvVO2vhG+zGLFzS/G0zC4
w+fw8a3W1o3JzTuttb/P0KXN0NDq7CPCz0W6xCrI1YXe5YLO1fr+/pLAxW26
wK3a3kK1xZLIzYvP1W68wyivulrO3jWwuhiwvTy4wkfJ0zDAzOfz9BGuu5Tf
5ZzZ3pTW1iq2wn3c5KvIy5fZ3cLm6bvZ2yKuulLF1oPY30q9znXDyX7N093v
8Ia/xEHDznjBxqXW3sXe5lPR2znDzzqwuszMzAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAACH5BAUUAKkALAAAAAAgACAAAAj/AFMJHEiwoMGD
CBH+WMhwYcKHAhv+AAQogUSIBhtWidTiwSISIrhEqtAQYyqGFSZ06LAGQYYM
iEjAgTOFBUOICwF9ksPzwRAPCz0gqNFmZhyLPxLmnFCmqY05gxpmINGmaIQz
QJNmnMjUhFNKAaRStbriQdatf8qYMOJ1gZAQDD2gobFkZoQVKyQ4LPijQhS1
JgKXOUUkRIBBQ9Ys2dOGxt0gQajsjfiDjVq2gRuJ2qACDZoRDzowdpwGQBAV
k09W+QvYRCMujIbcIQAEzZwzZzrQqJFGDQAAerSe/DGGtYkpB1oU0RMg7EIK
QhydeSCCxCJIkKzsXTjhL9spLlpM/wIlUQujHacePFizyAIk1FoX2vh7fIqp
CVCQZGqYZwgXUTsIQYkbFliwyHY/9BFFfWH4wMYHSmzB3xtQQMHFBnOMoImB
CPahln0OkiEFDj00BEYOcYAAhQaMEIHIEwfG98MCapkSBhsiosBAiTkVIEgW
OIAgBh+daTICgkyBKGIlR/zBQ0MFCJABAzhIIMYQiiAwAIJjlDHFjVJU4kcJ
hpjR0AyhMPFJFnFciQoCMUz2QxXHORhmCVfE4MSZPcRgSAY45DAAKoqcINxw
Okzhg4hHRJKCFnsy1AUPMcSQgScSvDEEDqktxMIBi8Lgh6NaWNLQpFrEwIQn
Smga1lYYTI1AhgFHXCGJFwo0hIcDHKjK6huhpEZZAjB8AAMnTcQQSBcNYWFJ
IC/c0IASh2AhLGVYGHAsBhBIxNAFdoTQwCjWHsrXDwlQYIAVMdSRhyVOKKCA
E3n0cEMIXpSL00KgfPGHIRwE7AUHhdhhRwxJ3IRRQ49gUsQlEF8CASZ0lGTS
QN5mfPFBGZu78ccIBQQAOw==
}]

set icon(question) [image create photo ::Message::icon::question -data {
R0lGODlhIAAgAPcAAAqAp8jo84Cotlakvi2iyKXM2fX5+zKPrp/T5Vq93iqq
1ZOwuh+GqfHx8YTL4kev0nG/2XOuwcXU2ZDB0k2jv4LA1QmRvjKbvZW/zWSw
yr7T2p3P4Nrj5iOVvEOw1W291wiMuIS8z3TD3Z7Dz02qyUSbuLXP2EWkxWSk
uVG22DuQq3G2zpTP4hqRuSKQtPf399zt8qe5wMnc4jmu1bjW4Fi11IrA0p7I
1h2MsYa2xnPG46bFzzuhw9fl6mSsxP///+Dy+VSkv5G3w3SzxzmdviqbwBiF
qWPA3x+Os1G42/D2923D3s3b30q223O93oXN5RSIr1GpxkKUtYnI3GutvS6W
udjv98DN0g1+pK7GzpXI2kS02qTQ3nrI44XD2Y68zTyqzz+oyw2RvQ2PuyuM
rJy9znq3zD2y2Ofw9EikwxiTvSmUuHyzxq7O2Fqrxmq0zb/d6IvF2HrE3Xq6
0Oft7zKews7n7qe6wFWy0DCjyTCs1qC/yZu0vZXQ5FS73Z7Y7I23xiiVumm/
3Pr9/oevvI3N4iqdw83g5kKevJPF1kO02ZvR4zmgw8Td5aPI1Ey12cPX3tzo
7WO41Tyozdbn7YvO5WitxHOwxMLP00uw0X3K5ITI4HvB2AmFrh+LsDyXtBWP
t97395e9yXu93g+LtU6YsbXFzkWtz6G8xjClzMzMzAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAACH5BAUUAKoALAAAAAAgACAAAAj/AFUJHEiwoMGD
CBH+WMhwYcKHAhcOevGigcUGFAf9gGiQ4YsecUoggQIFx6cKh15o3MhRoh0K
LobQaNiIDZITcFZClGimCAVKDRsqsdTJjRKHCn8MwsMoyNGgQS2RKvL0oMQa
kxABZahERqOnC5WQAWGoasGFFcDwGNGwzQEXZMgUaFigkwU3SAn+sDJjEhE0
DGUcqJOHQBEGdBjCAADCApy8qhYe6Ruk4SVGYM7MSLXmS0MspCwwgvwDiJ4Z
YN5wRXRqy5Yzk6p4ZghazBg7LCP/+KPnDJgwPEogSpMpRZJHD8IckMGwkRFQ
Y8Z4QbqwxowzZzw8yJSpRoLvNUhI/zHRMIKnFi3GIKL+4/QZRU2a+Elw5Igg
SW4ibF3YhsyaDuipwZ4C18U3X31LfJABBmD9YIIKRFwQCIACsrQQgVvIR98S
OoiwAlsN7VAKBWnwIGELFUb0w3VbJOHHERx2wQkbBjBEByEo+OBGFCcQUUUg
gbBXwxlNJJEAh5o4UAGIC/ERgRkrvLEjIhesQQF7fxD54hJdPFGIDRo0JAAg
cVSwQgZRUOBjInmVdoaRR+igSSUsJAJJQ0IkooUXc0hJQQlV9JCbbh88cqQO
DvSxyA1tSHCFQCMUsEEcIQzhAwWIqDboQgEYKogcT/SBABcjLHBHDHxEWkAi
IazgQxCIHJQyqIpxJPBpoggUsEdiPyiBiiM3aFHpqzlApqIBEAgiwiYsbDAC
Ew0xMcINrFoagVln/WAAJxBsUogWGETSEAeiJJLIHEMM8YKxemm7ASdTaDGB
BA1lgUEiNswxwrqzdrQQDAhMEQcgkDRARxaATDBBIuKyaxVDaMBxwxc55PDF
DY0AxhBH2SpFUUYNcZwUVCKXXHJAADs=
}]

set icon(error) [image create photo ::Message::icon::error -data {
R0lGODlhIAAgAPcAAKWMD+njwLSna6iXQOHMZ86uEf79+tC4P6qWMNzSo8e6
c+zq47+rRc60MbOaIcq3Vce+keHcxtfMnK+meriqZPPsyLudCNKzHKiaWNXF
eL2lM76uW9bKkNS+Ub2iINrAQ9e7M/j28Oro3r2vbca6gr2iGb+0fN3QjO7r
3rGeRLmcD+rdmdXAWr6sT82wIejm2Ne/Q8SrMbalVNS4MeTZps3DlLqzkNW3
KcKmIrihMKyRCPf39djKhN/by9XJjrygEvHs2KmSJs+yG8i1Wcu5YL6nO9nE
XLynQr21a9zETMO0ZbWUEL2qSt7TnNe9O7aqdObepcKlENC/bN7Slcu9erOk
W9nASNzEUevmzdC3OsK4hLOdMOvn1ubfvc/CgtzGWaqdYvn038G0dqycS8Kv
U7WlKcq2TNvLffTx5bWref///+3q362jcrmxh+jZj7ajQ7idF9a1OsyuF86z
KcSqLLGaKd3Qk9C8W72tMdTIksGqNrijO8OuSK2aPcK7mdzRna+fUrygGc++
ZtfEZNO1IbygHuDVpcWtO7utZNXHhb2jJsWvQN69SuPZqrOXB+bcq9nCTdG7
TubhzOPexMWtEMW1UrOfObuoS/Hv5cS6iezenNG0IN3Ohvfx1sW1a861WrKg
SL21lLGXFtPBa7enX8W2cLiugLOocr+1gczMzAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAACH5BAUUAKkALAAAAAAgACAAAAj/AFMJHEiwoMGD
CBGqWchwYcKHAhtKbAjRYMMwKyLdECKE0AE3nShWZNiJxY2ThAhduMCxQJIK
DCEuNJBhhs0ZKFVylFOAEpEQDhWqMcAiS5YGWWzmZCmHUpQodICqOTiz6KEh
Q2I0uIHzBqFNQkoUGuLIQtSgBBcmynJoQ4gdLfQ06HrDRYk6XNQo0WFhCNqB
ajqBuCp1hwytNufgwMtQSdkAfxcSGIylYYgNchvQ2ZKXYYKyeiKHAWH0CBrL
G4posNR54SQASyxYqAxYjSYQTthu2GG5yoDWargEEfVDhQoeQReywO3kwCK3
ll805ILAQaFAcFQUSa5mBu4PMCIt/xohdeKLPjnoKPIAB04J7je+W4HUodII
3hJfjHlzSI/6QnD8AB8IMFiRxBVf3CFDGhOxMQYTfCyiRw4eXMedfAgaQQQF
Iky0wykykMHHIRqU4YAH3LHgxAcHajgCChMttMMTMrTAQBE5KHIJd5qsCAkL
dyCyRkML4CfjKYBcckQOZXCAlhphOGFFBw8gAmNME9hgpBof7rdHDnmlpcYg
4W1wpUNa1ACBH1vuIEAKexARmRoVwFBJZTFlIoEhCdTgxwINSTLGHpJMVZBa
ZpSCXwRaSEDDI4302QagaoiAASio/BURl4KQIcYOEZjgQxOPQGrIHxCYssAC
YACCiFRUqWARghRDUKFAIlM0QcOup+ZBgglPkEKepmLukMcoGfBwQq5NGNLE
FD54oYAnEOAn00JATHEGJyfYMcUUdnCQSAZ5tFZRKg2hEUATJ5zArQRdYCLS
uZvGOC+9Yk6E777nBgQAOw==
}]

set icon(warning) [image create photo ::Message::icon::waring -data {
R0lGODlhIAAgAOYAANrGG+/v79nPhdbFU+/ow+ndg+3WOe/bUd7avP///+XV
VePOLOnceOHWhuncdc+5OPjqjvfWOunbavf399/ZuMe9eObWWf/2xPfxveDX
kN7FOfn14uXPRP/mWs/KpvffSufZZvXpptTFaerem+7ZSfzfQ/DdZP798tDL
neDVmd/LKubk1uXXeOPJO/7pcs6/W/baQNrPeuvQP+7hhcu4P+LPOc/Hk9vV
qefUR+vlwP//993RjP3zt/HaQv/tie/jhu7jlNTARdzJJvz32PHbSOrVPNbE
XP/33vDrzPfdQ//xofTy5uvYUfDfa/HdXNnTo97Qbe/he97MM+fUQeLMUOnT
M+/fddK8PfjrkNXPnevZc869TOHQQ9jQl+3ozObMPO7YQPbcPP/40OPZleTZ
muzp2eTeu+jYg/v45Ofdjuzhi/3nZebWa9TOo97WjN7JINjPjfjrqO7jitbH
Y8zMzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAUUAHQA
LAAAAAAgACAAAAf/gHSCg4SFhoeIiYqLjHQJCY2Rgo+Pko2UkJaKmJWaiI8X
Yp2ehZQdHZSkpQlKJSVKo6qPGx8wMB8bsZqPE1otGhotBRO6kgkTOSNIXl5I
ZznFl8dUSJhkFsSZlo8hGl6YKS1x0ZsJJx/e4C0fJ+SHlD4w6ZQ7XyU+7oaP
YiUwLSmYRMhwJUpbuQRr+v3DZGRgCRepFj3iEaZEmBYiMD2QkcQVj3yOzH0I
E6bHl4yPllzhmCQMu3yPIEQo2YMDDUpegpAgkqRHGCzuHg0xYKAHESIKtlAy
M8DCgQNEeoAZEo2SCQNgSBxwYuGFDQ8eKswBYcLJARJgmkQk9AhDlSIklZg4
kQClS5kVK8rAYWGliQkmaDHoenQCR5UpciUwiEEBU5YGM6L4ZYK4ncFHQBbU
wKFAgpUCAhBgapMByA8rEizgKAJk1CM0b1TU4GBBQpQCO8xgujEGiJwoqXHU
WICm0yMFAIRI4aIAhIMCZ7pkQYHiyY40ago4AKGAixQhCipxGn9sQoDzAbKR
X8+ekvr2IFXJZxQIADs=
}]
}

proc Message::_listNames {names} {
	append elist [join [lrange $names 0 end-1] ", "]
	append elist ", or [lindex $names end]"
	return $elist
}

proc Message::_doButtonCommand {dialog script button} {
	set code [catch [uplevel #0 $script $button] result]
	if {$code == 1} {
    	return -code $code -errorinfo $::errorInfo -errorcode $::errorCode $result
	} elseif {$code == 3 || $code == 4} {
		return
	}
	destroy $dialog
}

#
# parse args and create dialogs
#
proc Message::_dialog {name args} {
	variable type
	variable icon
	variable label
	variable config

	array set options {
		-buttons ""
		-labels  ""
		-default ""
		-cancel  ""
		-icon    ""
		-image   ""
		-title   ""
		-message ""
		-detail  ""
		-textbox 0
		-command ""
		-parent  "#AUTO"
		-type    ""
		-buttonalign "right"
	}

	# set default dialog types
	foreach {option value} $args {
		if {$option ne "-type"} {continue}
		if {[array names type $value] eq ""} {
			error "bad type \"$value\": must be [_listNames [array names type]]"
		}
		array set options $type($value)
		break
	}

	# overwrite default options
	foreach {option value} $args {
		if {[array names options $option] eq ""} {
			error "bad option \"$option\": must be [_listNames [array names options]]"
		}
		set options($option) $value
	}

	# check options
	if {$options(-icon) ne "" && [array names icon $options(-icon)] eq ""} {
		error "bad icon \"$options(-icon)\": must be [_listNames [array names icon]]"
	}
	if {$options(-image) ne "" && [lsearch [image names] $options(-image)] < 0} {
		error "unknown image \"$options(-image)\""
	}
	if {$options(-parent) ne "#AUTO" && $options(-parent) ne ""  &&
		[winfo exists $options(-parent)] == 0} {
		error "unknown parent window \"$options(-parent)\""
	}
	if {[lsearch "left right center" $options(-buttonalign)] < 0} {
		error "bad buttonalign \"$options(-buttonalign)\": must be [_listNames {left right center}]"
	}

	# build dialog
	toplevel $name -class Dialog
	wm withdraw $name
	wm title $name $options(-title)
	wm protocol $name WM_DELETE_WINDOW ""

	if {$options(-parent) ne ""} {
		if {$options(-parent) eq "#AUTO"} {
			set parent [winfo toplevel [winfo parent $name]]
			while {[winfo class $parent] eq "Menu" && $parent ne "."} {
				set parent [winfo toplevel [winfo parent $parent]]
			}
		} else {
			set parent $options(-parent)
		}
		wm transient $name $parent
	}
	
	# base frame
	set f [ttk::frame $name.f]

	# command buttons
	if {[llength $options(-buttons)] > 0} {
		set cmd [ttk::frame $f.cmd]
		set buttons [ttk::frame $cmd.buttons]
		
		array set labels [array get label]
		array set labels $options(-labels)
		foreach b $options(-buttons) {
			if {$b eq ""} {break}
			if {[array names labels $b] ne ""} {
				set l $labels($b)
			} else {
				set l $b
			}
			ttk::button $buttons.$b -text $l -command \
			[namespace code "_doButtonCommand $name \{$options(-command)\} $b"]
			bind $buttons.$b <Return> [list event generate $buttons.$b <<Invoke>>]
		}
		eval pack [winfo children $buttons] -side left -padx \{ $config(interspace) 0 \}
		if {$options(-buttonalign) eq "center"} {
			set options(-buttonalign) top
		}
		pack $buttons -side $options(-buttonalign) -fill none -expand false

		if {$options(-default) ne ""} {
			if {[winfo exists $buttons.$options(-default)]} {
				$buttons.$options(-default) configure -default active 
				focus -force $buttons.$options(-default)
			}
		}
		if {$options(-cancel) ne ""} {
			set esc_script "event generate $buttons.$options(-cancel) <<Invoke>>"
			bind $name <Escape> $esc_script
			wm protocol $name WM_DELETE_WINDOW $esc_script
		}
	}

	
	set fm $name.f.msg
	if {$options(-image) ne "" || $options(-icon) ne "" ||
		$options(-message) ne "" || $options(-detail) ne ""} {
		ttk::frame $fm

		# icon or image
		if {$options(-image) ne ""} {
			ttk::label $fm.icon -image $options(-image)
		} elseif {$options(-icon) ne ""} {
			ttk::label $fm.icon -image $icon($options(-icon))
		}
		
		# message
		if {$options(-message) ne ""} {
			ttk::label $fm.message -text $options(-message) \
				-font TkCaptionFont -wraplength $config(textwidth)\
				-anchor w -justify left
		}

		# detail message
		if {!$options(-textbox) && $options(-detail) ne ""} {
			ttk::label $fm.detail -text $options(-detail) \
				-font TkTextFont -wraplength $config(textwidth) \
				-anchor w -justify left
		}
	}

	# text box
	if {$options(-textbox)} {
		ttk::frame $f.textbox -style TEntry -border 2
		set textfont $::SQLText::info(font)
		set texttab $::SQLText::info(tab)
		set textn [expr {72.0 / [winfo pixels . 1i]}]
		ttk::scrollbar $f.textbox.y -command [list $f.textbox.text yview]
		text $f.textbox.text -relief flat -undo no -highlightthickness 0 \
		-wrap word -background white -font $textfont -takefocus 1 \
		-tabs [expr {$textn * [font measure $textfont 0] * $texttab}] \
		-yscrollcommand [list $f.textbox.y set]
		$f.textbox.text insert 1.0 $options(-detail)
		$f.textbox.text mark set insert 1.0
		$f.textbox.text configure -state disabled

		set textpop [::Util::makeEditorPop $f.textbox]
		foreach mlabel {Cut Paste Delete} {
			$textpop entryconfigure [msgcat::mc $mlabel] -state disabled
		}
		bind $f.textbox.text <<Copy>> { tk_textCut %W }
		bind $f.textbox.text <<B3>> [subst {
			focus -force %W
			tk_popup $textpop %X %Y
			break
		}]
		bind $f.textbox.text <FocusIn>  [list + $f.textbox state focus]
		bind $f.textbox.text <FocusOut> [list + $f.textbox state !focus]

		grid $f.textbox.text -row 0 -column 0 -sticky news
		grid $f.textbox.y    -row 0 -column 1 -sticky ns
		grid columnconfigure $f.textbox 0 -weight 1
		grid rowconfigure    $f.textbox 0 -weight 1
		grid propagate $f.textbox 0

		$f.textbox configure -width $config(textwidth) -height 80
	}

	# put frames on dialog
	if {[winfo exists $f.cmd]} {
		pack $f.cmd -side bottom -expand false -fill x \
		-pady [list $config(margin) [expr {$config(margin) * 2}]] \
		-padx $config(margin)
	}
	if {[winfo exists $fm]} {
		if {[winfo exists $fm.icon]} {
			pack $fm.icon -side left -anchor n -expand false \
			-pady $config(margin) -padx $config(margin)
		}
		if {[winfo exists $fm.message]} {
			pack $fm.message -side top -expand false -fill x \
			-pady $config(margin) -padx $config(margin)
		}
		if {[winfo exists $fm.detail]} {
			pack $fm.detail -side top -expand false -fill x \
			-padx $config(margin)
		}
		pack $fm -side top -expand no -fill x -pady 0 -padx 0
	}
	if {[winfo exists $f.textbox]} {
		pack $f.textbox -side bottom -expand yes -fill both \
		-pady [list $config(margin) $config(margin)] \
		-padx $config(margin)
	}

	pack $f -expand true -fill both
	wm minsize $name [expr {int($config(textwidth)*0.7)}] 64
	update
	return $name
}
#
# define new type
#
proc Message::defineType {name args} {
	variable type
	set type($name) $args
}


proc Message::msgset {btn} {
	variable retval $btn
}

# return selected button name
proc Message::show {args} {
	variable retval
	set root .ttkmessage
	if [winfo exists $root] {
		destroy $root
	}

	eval _dialog $root $args -command {Message::msgset}
	wm withdraw $root
	set focus [focus]

	::tk::PlaceWindow $root widget .
	::tk::SetFocusGrab $root $root
	focus -force $focus
	tkwait window $root
	::tk::RestoreFocusGrab $root $root

	return $retval
}
;#>>>

Message::defineType apperror -default ok -buttons ok -textbox 1 \
-icon error -message "Error Message" -title "Error Message"
Message::defineType appinfo -default ok -buttons ok -textbox 1 \
-icon info -message "Information" -title "Information"

#
# OneLine GetString MessageBox
#
namespace eval GetString {;#<<<
	variable state {}
	variable retval {}
	variable root ._getstring
}

proc GetString::show {title message {defaultval {}} {parent .}} {
	variable state
	variable retval
	variable root
	if ![winfo exists $root] {
		toplevel $root
		ttk::frame $root.f
		ttk::label $root.f.msg -wraplength 400 -anchor w -justify left -text {}
		ttk::entry $root.f.entry -width 50 \
			-textvariable [namespace current]::retval
		ttk::frame $root.f.cmd
		ttk::button $root.f.cmd.ok -text OK
		ttk::button $root.f.cmd.cancel -text Cancel
		pack $root.f.cmd.cancel $root.f.cmd.ok -side right -fill x -padx 2
		pack $root.f.msg $root.f.entry $root.f.cmd -fill x -padx 4 -pady 4
		pack $root.f -expand yes -fill both
		foreach w [winfo children $root.f.cmd] {
			Util::bindMouseEvent $w
		}
		# bind
		set cancelAction [namespace code {wm withdraw $root;set state cancel}]
		set okAction [namespace code {wm withdraw $root;set state ok}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$root.f.cmd.ok     configure -command $okAction
        $root.f.cmd.cancel configure -command $cancelAction
		bind $root.f.entry <Return> $okAction
		Util::bindEditorPopCommon $root.f.entry
		wm resizable $root 0 0
	}
	# Init Value
	if {$defaultval ne ""} {
		set retval $defaultval
		$root.f.entry selection range 0 end
	} else {
		set retval ""
	}
	wm title $root $title
	$root.f.msg configure -text $message 
	set state show

	wm withdraw $root
#    lower $root .
#    ::Util::moveCenter $root
#    Util::ngrab set $root
#    wm transient $root $parent
#    wm deiconify $root
#    focus -force $root.f.entry
#    tkwait var [namespace current]::state
#    Util::ngrab release

	wm transient $root $parent
	::tk::PlaceWindow $root widget .
	::tk::SetFocusGrab $root $root
	::focus -force $root.f.entry
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root

	if {$state eq "ok"} {
		return $retval
	}
	return ""
}
;#>>>



#
# Parser for SQL statement of SQLite
#
# SQL Parse for SQLite
if {[info tclversion] < 8.5} {
	if {[info command dict] eq ""} {
		package require dict
	}
}

namespace eval SQLParser {}

proc SQLParser::token {str posName {passPiriod 1}} {
	upvar $posName pos
	set plist [regexp -start $pos -indices -inline -- {[^\s]} $str]
	if {[llength $plist] == 0} {
		set pos [string length $str]
		return
	}
	set pos [lindex $plist 0 0]
	set c [string index $str $pos]
	switch -exact -- $c {
		\[ {
			set ed [string first \] $str [expr {$pos+1}]]
			incr ed
		}
		\" -
		' {
			set ed [string first $c $str [expr {$pos+1}]]
            if {$ed == -1} {error "unknown token [string range $str $pos end]"}
            incr ed
			while { [string index $str $ed] eq $c } {
                incr ed
				set ed [string first $c $str $ed]
                if {$ed == -1} {error "unknown token [string range $str $pos end]"}
				incr ed
			}
		}
		default {
			set ed [string wordend $str $pos]
		}
	}

	set word [string range $str $pos [expr {$ed - 1}]]
	set pos $ed

	if {$passPiriod} {
		if {[string index $word end] eq "." ||
			[string index $str $pos] eq "."} {
			append word [token $str pos 1]
		}
	}
	# move to next head of word
	set plist [regexp -start $pos -indices -inline -- {[^\s]} $str]
	if {[llength $plist] == 0} {
		set pos [string length $str]
	} else {
		set pos [lindex $plist 0 0]
	}
	return $word
}

proc SQLParser::ptoken {str posName {passPiriod 1}} {
	upvar $posName pos
	if {$pos == 0} {return ""}

	incr pos -1
	while {$pos > 0} {
		set st [string wordstart $str $pos]
		if {[string is space [string index $str $st]]} {
			incr pos -1
		} else {
			if {$passPiriod} {
				if {[string index $str [expr {$st-1}]] eq "."} {
					incr pos -2
					continue
				}
			}
			set pos $st
			break
		}
	}
	set npos $pos
	return [token $str npos $passPiriod]
}
# pos isn't changed
proc SQLParser::getNextToken {str pos {passPiriod 1}} {
	return [token $str pos $passPiriod]
}
# pos isn't changed
proc SQLParser::getPrevToken {str pos {passPiriod 1}} {
	return [ptoken $str pos $passPiriod]
}

proc SQLParser::getList {str} {
	set pos 0
	set len [string length $str]
	set rlist [list]
	while {1} {
		set word [token $str pos]
		if {$word eq ","} {return}
		lappend rlist $word
		if {$pos == $len} {break}

		set word [token $str pos]
		if {$word ne ","} {return}
		if {$pos == $len} {return}
	}

	set r [dict create]
	set i 0
	foreach v $rlist {
		dict set r $i [unQuote $v]
		incr i
	}
	return [dict create value $r]
}

proc SQLParser::unQuote {str} {
	set q [string index $str 0]
	switch -exact -- $q {
		\" -
		' {
			if {[string index $str end] ne $q} {
				error "invalid quote"
			}
			return [string map [list $q$q $q] [string range $str 1 end-1]]
		}
		\[ {
			if {[string index $str end] ne {]}} {
				error "invalid quote"
			}
			return [string range $str 1 end-1]
		}
		default { return $str }
	}
}

proc SQLParser::removeComment {sql} {
	set sql [regsub -all -- {/\*.*?\*/} $sql " "]
	set sql [regsub -all -line -- {--.*$} $sql " "]
	return $sql
}

proc SQLParser::splitDB.Name {str} {
	set pos 0
	set tname1 [token $str pos 0]
	set period [token $str pos 0]
	set tname2 [token $str pos 0]
	if {$tname1 eq "" || $tname1 eq "."} {return}
	if {$period eq ""} {
		return [list db "" table [unQuote $tname1]]
	}
	if {$period eq "."} {
		if {$tname1 eq "" || $tname2 eq ""} {return}
		return [list db [unQuote $tname1] table [unQuote $tname2]]
	}
	return
}

proc SQLParser::parseCreateTable {sql} {
	set q {^\s*CREATE\s+(?:(TEMP|TEMPORARY)?\s*)TABLE\s+([\"'\[\]\w. ]*.)\s*\((.*)\)}
	regexp -nocase -- $q $sql all temp dbtable columns
	if {[info locals all] eq ""} {return}
	set sql $all
	set data [dict create]
	# set temp
	if {$temp ne ""} { dict set data temp 1 }
	# set db and table name
	set dbtable [splitDB.Name $dbtable]
	if {[dict size $dbtable] == 0} {
		return
	}
	set data [dict merge $data $dbtable]; #add db and table

	#set column defs
	set column [dict create]
	set len [string length $columns]
	set pos 0
	set i 0
	while {$pos <= $len} {
		set col [parseColumn $columns pos]
		if {$col eq ""} {break}
		dict set column $i $col
		incr i
	}
	dict set data column $column

	# set constraints defs
	set i 0
	set constraint [dict create]
	while {$pos <= $len} {
		set const [parseTableConstraint $columns pos]
		if {$const eq ""} {break}
		dict set constraint $i $const
		incr i
		if {[getNextToken $columns $pos] eq ","} {
			token $columns pos
		}
	}
	if {[dict size $constraint] > 0} {
		dict set data constraint $constraint
	}
	if {$len != $pos} { return }
	return $data
}

proc SQLParser::parseColumn {sql posName} {
	upvar $posName posOriginal
	set pos $posOriginal
	set constkeyword {NOT NULL PRIMARY UNIQUE CHECK CONSTRAINT
		DEFAULT COLLATE AUTOINCREMENT FOREIGN}
	set info [dict create]
	
	# Column Name
	set name [token $sql pos]
	if {[lsearch $constkeyword [string toupper $name]] != -1} {
		return
	}
	if {$name eq ""} {return}
	dict set info name [unQuote $name]
	
	# Column Type
	set tpos $pos
	set type [token $sql pos]
	if {[lsearch $constkeyword [string toupper $type]] != -1} {
		set word $type
	} else {
		if {$type eq "," || $type eq ""} {
			set posOriginal $pos
			return $info
		}
		set _pos $pos
		if {[token $sql _pos] eq "("} {
			set pos [string first ")" $sql $_pos]
			if {$pos < 0} {return}
			dict set info type [string range $sql $tpos $pos]
			token $sql pos
		} else {
			dict set info type $type
		}
		set word [token $sql pos]
	}
	# Column Constraints
	set constlist [dict create]
	set i 0
	while {$word ne "" && $word ne ","} {
		set const [dict create]
		set word [string toupper $word]

		if {$word eq "CONSTRAINT" || $word eq "CONSTRAINTS"} {
			set constname [token $sql pos]
			if {$constname eq "," || $constname eq ""} {return}
			dict set const name $constname
			set word [string toupper [token $sql pos]]
		}

		switch -exact -- $word {
			NULL { 
				# Do nothing. NULL is default.
				set word [token $sql pos]
				continue
			}
			NOT {
				if {[string toupper [token $sql pos]] ne "NULL"} {
					return
				}
				dict set const type notnull
				set conflict [parseConflict $sql pos]
				if {[dict size $conflict] > 0} {
					set const [dict merge $const $conflict]
				}
			}
			PRIMARY {
				if {[string toupper [token $sql pos]] ne "KEY"} {
					return
				}
				dict set const type pkey
				
				set _pos $pos
				set sort [token $sql _pos]
				if {[lsearch {ASC DESC} [string toupper $sort]] > -1} {
					dict set const sort [string tolower $sort]
					set pos $_pos
				}
				set conflict [parseConflict $sql pos]
				if {[dict size $conflict] > 0} {
					set const [dict merge $const $conflict]
				}
				
				set _pos $pos
				set autoincr [token $sql _pos]
				if {[string toupper $autoincr] eq "AUTOINCREMENT"} {
					dict set const autoincrement 1
					set pos $_pos
				}
			}
			UNIQUE {
				dict set const type unique
				set conflict [parseConflict $sql pos]
				if {[dict size $conflict] > 0} {
					set const [dict merge $const $conflict]
				}
			}
			CHECK {
				set exp [parseParenthesis $sql pos]
				if {[dict size $exp] == 0} {return}
				dict set const type check
				dict set const value [string trim [dict get $exp value]]
				
				set conflict [parseConflict $sql pos]
				if {[dict size $conflict] > 0} {
					set const [dict merge $const $conflict]
				}
			}
			DEFAULT {
				set dval [token $sql pos 1]
				if {$dval eq "," || $dval eq ""} {return}
				if {[string toupper $dval] ne "NULL"} {
					if {$dval eq "("} {
						ptoken $sql pos 1
						set dvaldict [parseParenthesis $sql pos]
						set dval [dict get $dvaldict value]
						if {$dval eq ""} {return}
						set dval "($dval)"
					} elseif {$dval eq "+" || $dval eq "-"} {
						set dnum [token $sql pos 1]
						if {![string is digit -strict $dnum]} {
							return
						}
						if {$dval eq "+"} {
							set dval $dnum
						} else {
							set dval -$dnum
						}
					}
					dict set const type default
					dict set const value [unQuote $dval]
				} else {
					set word NULL
					continue
				}
			}
			COLLATE {
				set cname [token $sql pos]
				if {$cname eq "," || $cname eq ""} {return}
				dict set const type collate
				dict set const value $cname
			}
			REFERENCES {
				ptoken $sql pos
				set fkey [parseFKeyDef $sql pos]
				if {[dict size $fkey] == 0} {return}
				dict set const type fkey
				set const [dict merge $const $fkey]
			}
			default {
				return
			}
		}
		dict set constlist $i $const
		incr i
		set word [token $sql pos]
	}

	if {[dict size $constlist] > 0} {
		dict set info constraint $constlist
	}

	set posOriginal $pos
	return $info
}

proc SQLParser::parseTableConstraint {sql posName} {
	upvar $posName posOriginal
	set pos $posOriginal

	set const [dict create]
	set word [string toupper [token $sql pos]]

	if {$word eq "CONSTRAINT"} {
		set constname [token $sql pos]
		if {$constname eq "," || $constname eq ""} {return}
		dict set const name $constname
		set word [string toupper [token $sql pos]]
	}

	switch -exact -- $word {
		CHECK   -
		UNIQUE  -
		PRIMARY {
			switch -exact -- $word {
				CHECK   { dict set const type check }
				UNIQUE  { dict set const type unique }
				PRIMARY {
					if {[string toupper [token $sql pos]] ne "KEY"} {return}
					dict set const type pkey
				}
			}

			set exp [parseParenthesis $sql pos]
			if {$word ne "CHECK"} {
				if {[dict size $exp] == 0} {return}
				set l [getList [dict get $exp value]]
				if {[dict size $l] == 0} {return}
				
				dict set const column [dict get $l value]
			} else {
				dict set const value [dict get $exp value]
			}
			
			# Conflict
			set conflict [parseConflict $sql pos]
			if {[dict size $conflict] > 0} {
				dict set const conflict [dict get $conflict conflict]
			}
		}
		FOREIGN {
			if {[string toupper [token $sql pos]] ne "KEY"} {return}
			dict set const type fkey
			set fcolstr [parseParenthesis $sql pos]
			if {[dict size $fcolstr] == 0} {return}

			set fromcolumn [dict get [getList [dict get $fcolstr value]] value]

			set fkey [parseFKeyDef $sql pos]
			if {[dict size $fkey] == 0} {return}
			if {[dict size [dict get $fkey tocolumn]] != [dict size $fromcolumn]} {
				return
			}
			dict set fkey fromcolumn $fromcolumn
			set const [dict merge $const $fkey]
		}
		default {return}
	}
	
	set posOriginal $pos
	return $const
}

proc SQLParser::parseFKeyDef {sql posName} {
	upvar $posName posOriginal

	set pos $posOriginal
	# REFERENCES reftable [ ( refcolumn ) ]
	# [ MATCH FULL | MATCH PARTIAL | MATCH SIMPLE ]
	# [ ON DELETE action ] [ ON UPDATE action ] [ ON INSERT action ]
	# [ DEFERRABLE | NOT DEFERRABLE [ INITIALLY DEFERRED | INITIALLY IMMEDIATE ]]
	set word [string toupper [token $sql pos]]
	if {$word ne "REFERENCES"} {return}
	set info [dict create]
	#table
	set tpos $pos
	set table [token $sql pos]
	if {$table eq "," || $table eq "(" || $table eq ""} {return}
	dict set info table $table
	#column
	set _pos $pos
	if {[token $sql _pos] eq "("} {
		set tocolumn [dict create]
		set i 0
		while {[set wc [token $sql _pos]] ne ")"} {
			if {$wc eq ""} {return}
			dict set tocolumn $i $wc
			switch -exact -- [getNextToken $sql $_pos] {
				{,} {token $sql _pos}
				{)} {}
				default {return}
			}
			incr i
		}
		dict set info tocolumn $tocolumn
		set pos $_pos
	}
	
	set prepos $pos
	set word [string toupper [token $sql pos]]
	while {$word ne ""} {
		switch -exact -- $word {
			MATCH {
				set match [token $sql pos]
				if {$match eq "" || $match eq ","} {return}
				dict set info match $match
			}
			ON {
				switch -exact -- [string toupper [token $sql pos]] {
					DELETE {set act ondelete}
					UPDATE {set act onupdate}
					INSERT {set act oninsert}
					default {return}
				}
				switch -exact -- [string toupper [token $sql pos]] {
					SET {
						switch -exact -- [string toupper [token $sql pos]] {
							NULL    {set acttype setnull}
							DEFAULT {set acttype setdefault}
							default {return}
						}
					}
					CASCADE  {set acttype cascade}
					RESTRICT {set acttype restrict}
					default  {return}
				}
				dict set info $act $acttype
			}
			NOT -
			DEFERRABLE {
				if {$word eq "NOT"} {
					if {[string toupper [getNextToken $sql $pos]] ne "DEFERRABLE"} {
						#TODO:not null check
						set pos $prepos
						break
					}
					token $sql pos
					set type notdeferrable 
				} else {
					set type deferrable
				}

				set initially ""
				if {[string toupper [getNextToken $sql $pos]] eq "INITIALLY"} {
					token $sql pos
					switch -exact -- [string toupper [token $sql pos]] {
						DEFERRED  {set initially deferred}
						IMMEDIATE {set initially immediate}
						default {return}
					}
				}
				dict set info $type $initially
			}
			"," -
			default {
				set pos $prepos
				break
			}
		}
		set prepos $pos
		set word [string toupper [token $sql pos]]
	}

	set posOriginal $pos
	return $info
}

proc SQLParser::parseConflict {sql posName} {
	upvar $posName posOriginal
	set pos $posOriginal
	
	set word [token $sql pos]
	lappend word [token $sql pos]
	if {[string toupper $word] ne {ON CONFLICT}} {
		return
	}
	set word [string toupper [token $sql pos]]
	if {$word eq ""} {return}
	
	if {[lsearch {ROLLBACK ABORT FAIL IGNORE REPLACE} $word] < 0} {
		return
	}
	set posOriginal $pos
	return [dict create conflict [string tolower $word]]
}

proc SQLParser::parseParenthesis {sql posName} {
	upvar $posName posOriginal 
	set pos $posOriginal
	
	set word [token $sql pos]
	if {$word ne "("} {return}
	set len [string length $sql]
	
	set brstack 0
	set len [string length $sql]
	set st [expr {[string last "(" $sql $pos] + 1}]
	while {$pos <= $len} {
		if {$word eq "("} {
			incr brstack
		}
		if {$word eq ")"} {
			incr brstack -1
			if {$brstack <= 0} { break }
		}
		set word [token $sql pos 1]
		if {$word eq ""} {return}
	}
	set posOriginal $pos
	set val [string range $sql $st [expr {[string last ) $sql $pos] -1}]]
	return [dict create value $val]
}

proc SQLParser::parseCreateIndex2 {sql} {
	return [SQLParser::parseCreateIndex $sql 2]
}

proc SQLParser::parseCreateIndex3 {sql} {
	return [SQLParser::parseCreateIndex $sql 3]
}

proc SQLParser::parseCreateIndex {sql version} {
	set q {^\s*CREATE\s+(?:(TEMP|TEMPORARY)?\s*)(?:(UNIQUE)?\s*)INDEX\s+(?:(IF\s+NOT\s+EXISTS)?\s*)(.*)}
	regexp -nocase -- $q $sql all temp unique ifnotexists query
	if {[info locals all] eq ""} {return}

	set data [dict create]

	if {$unique eq ""} {
		dict set data unique 0
	} else {
		dict set data unique 1
	}

	if {$temp eq ""} {
		dict set data temp 0
	} else {
		dict set data temp 1
	}

	if {$ifnotexists eq ""} {
		dict set data ifnotexists 0
    } else {
		dict set data ifnotexists 1
    }
    
	set pos 0

    
	if {$version == 2} {
		dict set data index [unQuote [token $query pos]]
	} else {
		set dbtable [splitDB.Name [token $query pos]]
		if {[dict size $dbtable] == 0} {
			return
		}
		dict set data db [dict get $dbtable db]
		dict set data index [dict get $dbtable table]
	}
	
	if {[string compare -nocase [token $query pos] "ON"] != 0} {
		return
	}

	if {$version == 2} {
		set dbtable [splitDB.Name [token $query pos]]
		if {[dict size $dbtable] == 0} {
			return
		}
		set data [dict merge $data $dbtable]; #add db and table
	} else {
		dict set data table [unQuote [token $query pos]]
	}
	
	if {[token $query pos] ne "("} {return}
	set i 0
	set columns ""
	while {1} {
		set wc [token $query pos]
		if {$wc eq ""} {return}
		if {$wc eq ","} {return}
		if {$wc eq ")"} {break}
		set name [unQuote $wc]
		
		set wc [token $query pos]
		if {[string compare -nocase $wc "COLLATE"] == 0} {
			set collate [token $query pos]
			set wc [token $query pos]
		} else {
			set collate ""
		}
		if {[string compare -nocase $wc "ASC"] == 0} {
			set sort "ASC"
			set wc [token $query pos]
		} elseif {[string compare -nocase $wc "DESC"] == 0} {
			set sort "DESC"
			set wc [token $query pos]
		} else {
			set sort ""
		}

		lappend columns $i [list name $name collate $collate sort $sort]
		incr i

		if {$wc eq ","} {continue}
		if {$wc eq ")"} {break}
		return
	}
	dict set data columns $columns

	set conflict [parseConflict $query pos]
	if {[dict size $conflict] > 0} {
		dict set data conflict [dict get $conflict conflict]
	} else {
		dict set data conflict ""
	}

    return $data
}

proc SQLParser::parseCreateTrigger {sql} {
	set q {^\s*CREATE\s+(?:(TEMP|TEMPORARY)?\s*)TRIGGER\s+(?:(IF\s+NOT\s+EXISTS)?\s*)(.*)}
	regexp -nocase -- $q $sql all temp ifnotexists query
	if {[info locals all] eq ""} {return}

	set data [dict create]
	if {$temp eq ""} {
		dict set data temp 0
	} else {
		dict set data temp 1
	}
	
	if {$ifnotexists eq ""} {
		dict set data ifnotexists 0
	} else {
		dict set data ifnotexists 1
	}
	
	set pos 0

	set event [dict create]
	
	dict set data triggername [unQuote [token $query pos]]

	set temp $pos
	switch -exact -- [string toupper [token $query pos]] {
		BEFORE {dict set event time "BEFORE"}
		AFTER  {dict set event time "AFTER"}
		INSTEAD {
			if {[string compare -nocase [token $query pos] "OF"] == 0} {
				dict set event time "INSTEAD OF"
			} else {
				return
			}
		}
		default {
			dict set event time ""
			set pos $temp
		}
	}

	dict set event columns ""
	switch -exact -- [string toupper [token $query pos]] {
		DELETE {dict set event type "DELETE"}
		INSERT {dict set event type "INSERT"}
		UPDATE {
			if {[string compare -nocase [getNextToken $query $pos] "OF"] == 0} {
				token $query pos
				set i 0
				set columns ""
				while {[set wc [token $query pos]] ne ""} {
					if {[string compare -nocase $wc "ON"] == 0} {
						ptoken $query pos
						break
					}
					if {$wc eq ","} {continue}
					lappend columns $i [unQuote $wc]
					incr i
				}
				dict set event type "UPDATE OF"
				dict set event columns $columns
			} else {
				dict set event type "UPDATE"
			}
		}
		default return
	}
	dict set data event $event
	
	if {[string compare -nocase [token $query pos] "ON"] != 0} {
		return
	}
	
	set dbtable [splitDB.Name [token $query pos]]
	if {[dict size $dbtable] == 0} {
		return
	}
	set data [dict merge $data $dbtable]; #add db and table

	dict set data foreach ""
	dict set data when ""
	dict set data steps ""

	set temp $pos
	if {[string compare -nocase [token $query pos] "FOR"] == 0} {
		if {[string compare -nocase [token $query pos] "EACH"] != 0} {return}
        set foreach [token $query pos]
		if {[string compare -nocase $foreach "ROW"] == 0} {
            dict set data foreach ROW
        } elseif {[string compare -nocase $foreach "STATEMENT"] == 0} {
            # FOR EACH STATEMENT is allowed only sqlite2.
            dict set data foreach STATEMENT
        } else {
            return
        }
	} else {
		set pos $temp
	}

	set temp $pos
	if {[string compare -nocase [token $query pos] "WHEN"] == 0} {
		set st $pos
		while {1} {
			set temp $pos
			set wc [token $query pos]
			if {$wc eq ""} {return}
			if {[string compare -nocase $wc "BEGIN"] == 0} {
				set pos $temp
				dict set data when [string trimright [string range $query $st [expr {$pos -1}]]]
				if {[dict get $data when] eq ""} {return}
				break
			}
		}
	} else {
		set pos $temp
	}

	if {[string compare -nocase [token $query pos] "BEGIN"] == 0} {
		set st $pos
		set pos [string length $query]
		while {1} {
			set wc [ptoken $query pos]
			if {$wc eq ";"} {continue}
			if {[string compare -nocase $wc "END"] == 0} {
				set ed [expr {$pos -1}]
				break
			}
			return
		}

		#TODO: split each steps.
		dict set data steps [string trim [string range $query $st $ed]]
	} else {
		return
	}
	return $data
}

proc SQLParser::parse {sql} {
	#dummy
	return
}


#
# Session Commands(Session::write/read)
#
if {$pref(usesession) != 0} {
namespace eval Session {;#<<<
	variable file .tcl_session ;# over write here

	proc write {session} {
		variable file
        if {![file exists [file dir $file]]} {
            if {[catch {file mkdir [file dir $file]} msg]} {
                puts "error:$msg"
                return
            }
        }

		if {[catch {open $file w} fp]} {
                puts "error:$fp"
                return
            return
        }
		fconfigure $fp -encoding utf-8
		puts $fp $session
		close $fp
	}

	proc read {} {
		variable file
		# remove cd open file fconfigure command from unsafeCommand. 
		# FIXME: puts should be added to the list.
		set unsafeCommand {
		exec exit glob load pwd socket source vwait
		fcopy proc
		}
		
		if {![file exists $file]} return
		if {[catch {open $file r} fp]} { error $fp }
		fconfigure $fp -encoding utf-8
		set session [::read $fp]
		close $fp
		unset fp
		
		foreach cmd $unsafeCommand {
			interp hide {} $cmd
		}
		
		set error ""
		if {[catch {eval $session} ret]} {
			append error $ret\n
		}
		
		foreach cmd [interp hidden {}] {
			interp expose {} $cmd
		}
		
		if {$error eq {}} return 
		error $error
	}
}
;#>>>

set Session::file $::pref(sessionfile)
}

#
# Define Objects
#
namespace eval Titlebar {};#<<<

proc Titlebar::Titlebar {wname text image} {
	set root [ttk::frame $wname -relief groove -padding 2]
	ttk::label $root.label -text $text -image $image -compound left
	ttk::frame $root.cmd 
	pack $root.cmd -side right
	pack $root.label -side left
	return $root
}

proc Titlebar::_getUniqButtonName {w} {
	set last [lindex [winfo children $w.cmd] end]
	if {$last ne {}} {
		set num [string map [list "$w.cmd.btn" {}] $last]
		incr num
	} else {
		set num 0
	}
	return $w.cmd.btn$num
}

# Create and Add Button into the titlebar
proc Titlebar::Button {w image text command} {
	set name [_getUniqButtonName $w]
	
	# FIXME:Tile Toolbutton borderwidth is 2 in winnative theme.
	# I don't know how to change it to 1 on tile.
	if {$tile_currentTheme eq "winnative"} {
		button $name -command $command \
			-image $image -relief flat -border 1 -takefocus 0 
		bind $name <Enter> {+ %W configure -relief ridge}
		bind $name <Leave> {+ %W configure -relief flat}
	} else {
		ttk::button $name -command $command -image $image \
			-takefocus 0 -style Toolbutton
	}
	pack $name -side right

	# add tooltip
	Tooltip::bindWin $name $text

	return $name
}

proc Titlebar::Separator {w} {
	set name [_getUniqButtonName $w]
	ttk::separator $name -orient vertical
	pack $name -side right -fill y -pady 2 -padx 2
	return $name
}

# Create and Add RadioButton into the titlebar
proc Titlebar::Radiobutton {w image text value varName command} {
	set name [_getUniqButtonName $w]

	# FIXME:Tile Toolbutton borderwidth is 2 in winnative theme.
	# I don't know how to change it to 1 on tile.
	if {$tile_currentTheme eq "winnative"} {
		radiobutton $name \
			-variable $varName \
			-value $value \
			-image $image \
			-command $command \
			-indicatoron false \
			-selectcolor {} \
			-width 14 -height 14 \
			-compound center \
			-border 1 \
			-relief flat \
			-overrelief ridge \
			-offrelief flat \
			-takefocus 0 
	} else {
		ttk::radiobutton $name \
			-variable $varName \
			-value $value \
			-image $image \
			-compound image \
			-command $command \
			-takefocus 0 \
			-style Toolbutton
	}
	pack $name -side right

	# add tooltip
	Tooltip::bindWin $name $text

	return $name
}
;#>>>


namespace eval RecentFile {;#<<<
	variable list [list]
	variable info
	array set info {}
	set info(max) 20
}

proc RecentFile::RecentFile {files} {
	return [menu .recentFile -postcommand {RecentFile::update}]
}

proc RecentFile::append {file} {
	variable list
	variable info
	
	set i 1
	set newlist [list $file]
	foreach e $list {
		if {$e ne $file} {
			lappend newlist $e
		}
		if {[incr i] >= $info(max)} break
	}
	set list $newlist
}

proc RecentFile::update {} {
	variable list
	variable info
	.recentFile delete 0 end
	set i 1
	foreach file $list {
		.recentFile add command -label $file \
		-command "Cmd::openDB [list $file]"
		if {[incr i] >= $info(max)} break
	}
}

proc RecentFile::dump {} {
	variable list
	variable info
	set session "#::[info level [info level]]\n"
	::append session "set [namespace current]::list \[list $list\]\n"
	::append session "array set [namespace current]::info \[list [array get info]\]\n"
	return $session
}
;#>>>


namespace eval Statusbar {;#<<<
	variable var
	array set var {
		encoding {}
		version {}
		time {}
		row {}
	}
}
proc Statusbar::Statusbar {} {
	ttk::frame .statusbar
	ttk::label .statusbar.enc -textvariable [namespace current]::var(encoding)
	ttk::label .statusbar.ver -textvariable [namespace current]::var(version)
	ttk::label .statusbar.time -textvariable [namespace current]::var(time)
	ttk::label .statusbar.row -textvariable [namespace current]::var(row)
	ttk::label .statusbar.dummy
	foreach w [winfo children .statusbar] {
		$w configure -takefocus 0 -relief flat -anchor w -padding 0
	}
	foreach n {0 1 2 3} {
		ttk::separator .statusbar.sep$n -orient vertical
	}
	if {[tk windowingsystem] eq "aqua"} {
	ttk::label .statusbar.sizegrip -takefocus 0 -width 2
	} elseif {[::Util::versionCompare "0.7.8" [package present tile]] > 0} {
	ttk::label .statusbar.sizegrip -takefocus 0 -width 0
	} else {
	ttk::sizegrip .statusbar.sizegrip -takefocus 0
	}
	pack .statusbar.sizegrip -side right -pady {2 0}
	pack .statusbar.ver .statusbar.sep0 .statusbar.enc .statusbar.sep1 \
	 -fill y -side left -pady {2 0} -padx 1

	pack .statusbar.time .statusbar.sep2 .statusbar.row .statusbar.sep3 \
	-fill y -side right -pady {2 0} -padx 1
	
	return .statusbar 
}

proc Statusbar::clear {} {
	variable var
	set var(encoding) "Encoding : unknown"
	set var(version) "Version : unknown"
	set var(time) "Time : 0 msec"
	set var(row) "Row : 0 line"
}

proc Statusbar::update {} {
	variable var
	set internal_enc ""
	if {$::database(name) ne ""} {
		if {$::database(version) == 3} {
			::Sqlite::evalQuery {PRAGMA encoding}
			set internal_enc \ ($::Sqlite::data)
		}
	}
	set var(encoding) "Encoding : $::database(encoding)$internal_enc"
	set var(version) "Version : $::database(version)"
	set var(time) "Time : [::Cmd::getSqlTime] msec"
	set var(row) "Row : [::Cmd::getTableRowCount] line"
}

;#>>>


namespace eval Dbtree {;#<<<
	variable tree
	variable pressed {{} {}}
}

# return next item recursively if item is opened
proc Dbtree::getGlobalNext {w item} {
	if {$item eq {} || $item eq "#0"} {
		return [lindex [$w children {}] 0]
	}
	if {[$w item $item -open]} {
		set children [$w children $item]
		if {$children ne {}} {
			return [lindex $children 0]
		}
	}
	set next [$w next $item]
	if {$next ne {}} {
		return $next
	}
	
	while 1 {
		set parent [$w parent $item]
		set pnext [$w next $parent]
		if {$pnext ne {}} { return $pnext }
		if {$parent eq {}} { return {} } ;# end item
		set item $parent
	}
	return {}
}

# select items from the point.
#FIXME: need refactoring
proc Dbtree::selectItemsFromPoint {w stpoint edpoint} {
	if {[lindex $stpoint 1] > [lindex $edpoint 1]} {
		set st [lindex [if 1 {$w identify [lindex $edpoint 0] [lindex $edpoint 1]}] 1]
		set ed [lindex [if 1 {$w identify [lindex $stpoint 0] [lindex $stpoint 1]}] 1]
	} else {
		set ed [lindex [if 1 {$w identify [lindex $edpoint 0] [lindex $edpoint 1]}] 1]
		set st [lindex [if 1 {$w identify [lindex $stpoint 0] [lindex $stpoint 1]}] 1]
	}
	
	# Select One Item
	if {$st ne {} && $st eq $ed} {
		if {$st ne "#0"} {
			$w selection set $st
		}
		return
	}
	
	# Title isn't selected
	if {$st eq "#0"} { set st {} }
	if {$ed eq "#0"} { set ed {} }

	# Select All or Select None
	if {$st eq {} && $ed eq {}} {
		set haveItem 0
		set x [lindex $stpoint 0]
		set sy [lindex $stpoint 1]
		set ey [lindex $edpoint 1]
		#swap
		if {$ey<$sy} { set tmp $ey; set ey $sy; set sy $tmp }
		#search first item
		for {set y $sy} {$y <= $ey} {incr y 5} {
			if {[$w identify $x $y] ne {nothing}} {
				set haveItem 1
				break
			}
		}
		if {$haveItem} {
			#select all
			set st [lindex [$w children {}] 0]
			set ed {}
		} else {
			#select nothing
			$w selection set {}
			return
		}
	}
	
	# Select Range
	set slist $st
	set nitem $st
	while {1} {
		lappend slist [set nitem [getGlobalNext $w $nitem]]
		if {$nitem eq $ed || $nitem eq {}} {break} 
	}
	$w selection set $slist
}

# setup widget
proc Dbtree::Dbtree {} {
	variable tree .dbtree.f.tree

	ttk::frame .dbtree
	
	set title [Titlebar::Titlebar .dbtree.title [msgcat::mc Database] {}]
	pack $title -side top -fill x
	
	ttk::frame .dbtree.f -style TEntry -border 2
	ttk::scrollbar .dbtree.f.y  -command {.dbtree.f.tree yview}

	ttk::treeview .dbtree.f.tree -pad 1 -show {tree} \
		-yscrollcommand {.dbtree.f.y set}
	bind .dbtree.f.tree <<TreeviewSelect>> {::Cmd::updateProperty}
	bind .dbtree.f.tree <<TreeviewOpen>>   {::Dbtree::openItem %W}
	Cmd::bindTextFocusTtkEntry .dbtree.f.tree .dbtree.f 
	
	# Flatten relief
	style layout Tree2 {
		Tree2.label -sticky nswe
	}
	style configure Tree2 -relief flat -background white
	.dbtree.f.tree configure -style Tree2
	style configure Item -padding {5 0}
	
	# Menu
	menu .dbtree.pop
	.dbtree.pop add command -label [msgcat::mc "Show Create Table Statement"] \
	-command {
		Cmd::showSQLStatement [lindex [::Dbtree::getSelection] 0 2 5] 
	}
	.dbtree.pop add separator
	.dbtree.pop add command -label [msgcat::mc "Create Table"]... \
	-command {::GUICmd::TableBuilder::run create}
;#    .dbtree.pop add command -label [msgcat::mc "Create View"] -state disabled
	.dbtree.pop add command -label [msgcat::mc "Create Index"]... \
	-command {::GUICmd::CreateIndex::run}
	.dbtree.pop add command -label [msgcat::mc "Create Trigger"]... \
	-command {::GUICmd::CreateTrigger::run}
	.dbtree.pop add separator
	.dbtree.pop add command -label [msgcat::mc "Drop Table/View"]... -command {Cmd::dropTable}
;#    .dbtree.pop add command -label [msgcat::mc "Drop Index"]...
;#    .dbtree.pop add command -label [msgcat::mc "Drop Trigger"]
	.dbtree.pop add separator

	.dbtree.pop add command -label [msgcat::mc "Modify Table Schema"]... \
	-command {::GUICmd::TableBuilder::run modify}

	# Define additional behavior of tree
	bind .dbtree.f.tree <ButtonPress-1> [namespace code {
		set pressed [list %x %y]
		if {[lindex [%W identify %x %y] 0] eq {nothing}} {
			%W selection set {}
		}
	}]

	bind .dbtree.f.tree <Control-B1-Motion> {break}
	bind .dbtree.f.tree <B1-Motion> [namespace code {
		selectItemsFromPoint %W $pressed [list %x %y]
	}]

	bind .dbtree.f.tree <<B3>> [namespace code {
		set pressed [list %x %y]
		if {[lindex [%W identify %x %y] 0] eq {nothing}} {
			%W selection set {}
			break
		}
	}]

	bind .dbtree.f.tree <Control-B3-Motion> {break}
	bind .dbtree.f.tree <B3-Motion> [namespace code {
		selectItemsFromPoint %W $pressed [list %x %y]
	}]

	bind .dbtree.f.tree <<B3>> {
		focus -force %W
		tk_popup .dbtree.pop %X %Y
		break
	}

	pack .dbtree.f.y -side right -fill y
	pack .dbtree.f.tree -expand yes -fill both
	pack .dbtree.f -expand yes -fill both

	return .dbtree
}


proc Dbtree::openItem {tree} {
	set node [lindex [$tree selection] 0]
	set data [$tree item $node -values]

	if {$data eq {}} return
	dputs {open  treeitem [$tree item $node]}
	dputs {data is $data}
	
	set db   [lindex $data 0]
	set type [lindex $data 1]
	set name [lindex $data 2]

	switch -exact -- $type {
		table {::Cmd::editTable $db $name}
		view  {::Cmd::showView  $db $name}
	}
}

proc Dbtree::clear {tree} {
	$tree delete [$tree children {}]
}


#TODO: rewrite using dict.
# return ?{name img {db type name tbl_name rootpage sql}} ...?
proc Dbtree::getSelection {} {
	variable tree
	set ret [list]
	foreach item [$tree selection] {
		lappend ret [list \
			[$tree item $item -text] \
			[$tree item $item -image] \
			[$tree item $item -values] \
		]
	}
	return $ret
}


# return selected items root db list
proc Dbtree::getSelectedDBName {} {
	variable tree
	array set dblist {}
	set selection [$tree selection]
	while {[llength $selection] > 0} {
		set temp [list]
		foreach item $selection {
			set parent [$tree parent $item]
			if {$parent ne {}} {
				lappend temp $parent
			} else {
				set dblist([$tree item $item -text]) {}
			}
		}
		set selection $temp
	}
	return [array names dblist]
}


proc Dbtree::update {} {
	variable tree
	set selected [$tree selection]
	::Dbtree::clear $tree
	if {$::database(name) eq {}} return
	
	# get all database
	set query {PRAGMA database_list;}
	if {[Sqlite::evalQuery $query] != 0} {return}

	# NOTE:treeview's id should not have white space?
	set dn 0
	set dblist [list]
	foreach {seq name file} $::Sqlite::data {
		set id $dn-[string map {{ } {_}} $name]
		$tree insert {} end -id $id -text $name -image $::img(db) \
		-values [list $seq $name $file] -open yes
		lappend dblist $name $id
		incr dn
	}
	
	# set table and view
	foreach {db dbid} $dblist {
		foreach {ttype} {table view} {
			set master [Sqlite::getSqliteMasterName $db]
			set query "select type, name, tbl_name, rootpage, sql from $master where type = '$ttype'"
			if {[Sqlite::evalQuery $query] == -1} {return}
			if {[llength $::Sqlite::data] > 0} {
				set folderid "$dbid.${ttype}Foler"
				$tree insert $dbid end -id $folderid \
				-text [msgcat::mc [string totitle $ttype]] \
				-image $::img(folder) -open yes
				
				# Sort by name
				set templist [list]
				foreach {type name tbl_name rootpage sql} $::Sqlite::data {
					lappend templist [list $type $name $tbl_name $rootpage $sql]
				}
				foreach tempdata [lsort -dictionary -index 1 $templist] {
					set in 0
					foreach {type name tbl_name rootpage sql} $tempdata {
						# Append item
						set id $in.$folderid.[string map {{ } {_}} $name]
						$tree insert $folderid end -id $id \
						-image $::img($ttype) -text $name \
						-values [list $db $type $name $tbl_name $rootpage $sql]
						incr in
					}
				}
			}
		}
	}
	
	# restore selected item
	foreach item $selected {
		if {[$tree exists $item]} {
			$tree selection add $item
		}
	}
}

;#>>>


namespace eval Property {;#<<<
	array set tab {}
	array set field {
		0,0	{}
		0,1	Column
		0,2 Type
		0,3 NN
		0,4 Default
	}
	array set index {
		0,0 Name
		0,1 Unique
		0,2 Column
		0,3 Db
		0,4 Table
	};# Db and Table is hidden
	array set trigger {
		0,0 Name
		0,1 SQL
		0,2 Db
	}
}

# setup widget
proc Property::Property {} {
	variable tab
	ttk::frame .property
	
	ttk::label .property.title -compound left -anchor w -image $::img(dummy16x16)
	grid .property.title -row 1 -column 0 -sticky ew

	ttk::notebook .property.notebook
	dputs {.property.notebook configure}
	set tab(field)   [ttk::frame .property.notebook.field]
	set tab(index)   [ttk::frame .property.notebook.index]
	set tab(trigger) [ttk::frame .property.notebook.trigger]
	.property.notebook add $tab(field)   -text Field
	.property.notebook add $tab(index)   -text Index
	.property.notebook add $tab(trigger) -text Trigger
	
	#
	# Field
	#
	$tab(field) configure -style TEntry -border 2
	ttk::scrollbar $tab(field).y -command "$tab(field).table yview"
	table $tab(field).table -rows 0 -cols 5 \
		-titlerows 1 -titlecols 1 -selecttitle 0 \
		-variable ::Property::field \
		-selecttype row -colstretch fill\
		-yscrollcommand "$tab(field).y set"
	$tab(field).table width 0 2 ;# Primary key
	$tab(field).table width 1 8 ;# Column
	$tab(field).table width 2 6 ;# Type
	$tab(field).table width 3 3 ;# Not NULL
	$tab(field).table width 4 6 ;# Default
	::Cmd::changeTableLookAndFeel $tab(field).table
	
	pack $tab(field).y -side right -fill y
	pack $tab(field).table -expand yes -fill both

	$tab(field).table tag configure null -image $::img(nullValue)
	Cmd::bindTextFocusTtkEntry $tab(field).table $tab(field)
	#
	# Index
	#
	$tab(index) configure -style TEntry -border 2
	ttk::scrollbar $tab(index).y -command "$tab(index).table yview"
	table $tab(index).table -rows 0 -cols 3 \
		-titlerows 1 -titlecols 0 -selecttitle 0 \
		-selecttype row -variable ::Property::index \
		-colstretch fill\
		-yscrollcommand "$tab(index).y set"
	$tab(index).table width 0 8 ;# Name
	$tab(index).table width 1 4 ;# Unique
	$tab(index).table width 2 10 ;# Column
	::Cmd::changeTableLookAndFeel $tab(index).table
	
	pack $tab(index).y -side right -fill y
	pack $tab(index).table -expand yes -fill both

	menu $tab(index).pop
	$tab(index).pop add command -label [msgcat::mc "Show Create Index Statement"] -command {
	::Cmd::showSQLStatement [lindex [::Property::getSelectedIndex] 5]}
	$tab(index).pop add separator
	$tab(index).pop add command -label [msgcat::mc "Create Index"]... \
	-command {::GUICmd::CreateIndex::run}
	$tab(index).pop add command -label [msgcat::mc "Drop Index"]... \
	-command {::Cmd::dropIndex}
	Cmd::bindTextFocusTtkEntry $tab(index).table $tab(index)
	bind $tab(index).table <<B3>> \
	"focus -force %W; tk_popup $tab(index).pop %X %Y; break;"

	#
	# Trigger
	#
	$tab(trigger) configure -style TEntry -border 2
	ttk::scrollbar $tab(trigger).y -command "$tab(trigger).table yview"
	table $tab(trigger).table -rows 0 -cols 1 \
		-titlerows 1 -titlecols 0 -selecttitle 0 \
		-variable ::Property::trigger \
		-selecttype row -colstretch fill\
		-yscrollcommand "$tab(trigger).y set"
	$tab(trigger).table width 0 20  ;# Name
	$tab(trigger).table width 1 5 ;# Sql(Don't show)
	::Cmd::changeTableLookAndFeel $tab(trigger).table
	
	pack $tab(trigger).y -side right -fill y
	pack $tab(trigger).table -expand yes -fill both
	.property.notebook select $tab(field)

	menu $tab(trigger).pop
	$tab(trigger).pop add command -label [msgcat::mc "Show Create Trigger Statement"] \
	-command [namespace code {
		set _row [$tab(trigger).table index [lindex [$tab(trigger).table curselection] 0] row]
		if {$_row eq {} || $_row == 0} {break}
		::Cmd::showSQLStatement $trigger($_row,1)
		unset _row
	}]
	$tab(trigger).pop add separator
	$tab(trigger).pop add command -label [msgcat::mc "Create Trigger"]... \
	-command {::GUICmd::CreateTrigger::run}
	$tab(trigger).pop add command -label [msgcat::mc "Drop Trigger"]... \
	-command {::Cmd::dropTrigger}
	
	Cmd::bindTextFocusTtkEntry $tab(trigger).table $tab(trigger)
	bind $tab(trigger).table <<B3>> \
	"focus -force %W; tk_popup $tab(trigger).pop %X %Y; break;"
	
	# grid
	grid .property.notebook -row 2 -column 0 -sticky news
	grid propagate .property 0
	grid columnconfigure .property 0 -weight 1
	grid rowconfigure    .property 2 -weight 1
	return .property
}

proc Property::clear {} {
	variable tab
	.property.title configure -text {} -image $::img(dummy16x16)
	$tab(field).table configure -rows 0
	$tab(index).table configure -rows 0
	$tab(trigger).table configure -rows 0
	$tab(field).table tag cell {} 0,0
}

proc Property::update {tlist} {
	if {$::database(name) eq {} || $tlist eq {}} {clear; return}
	foreach {text image data} $tlist {break}

	# db == 3, folder == 0
	if {[llength $data] < 4} {
		clear; return
	}
	# change title
	.property.title configure -text $text -image $image
	
	set db   [lindex $data 0]
	set name [lindex $data 2]
	setField $db $name   ;#set field
	setIndex $db $name   ;#set index
	setTrigger $db $name ;#set trigger
}

proc Property::setField {db table} {
	variable field
	variable tab
	#element of table
	set colpkey 0
	set colname 1
	set coltype 2
	set colnull 3
	set coldef  4
	set clist [::Sqlite::getColumnList $db $table]

	$tab(field).table tag cell {} 0,0
	$tab(field).table selection clear all
	$tab(field).table configure -rows [expr {[llength $clist]+1}]

	set row 1
	foreach cdata $clist {
		dict with cdata {};#set values

		if {$null==0} {
			set null {}
		} else {
			set null NN
		}
		set field($row,$colname) $name
		set field($row,$colnull) $null
		set field($row,$coltype) $type
		set field($row,$coldef)  $def
		$tab(field).table tag cell [expr {$defisnull ? "null" : ""}] $row,$coldef
		
		if {$pkey == 1} {
			$tab(field).table tag cell pkey $row,$colpkey
		} else {
			$tab(field).table tag cell {} $row,$colpkey
		}
		incr row
	}
}

proc Property::setIndex {db table} {
	variable index
	variable tab
	set ilist [::Sqlite::getIndexList $db $table]
	
	$tab(index).table selection clear all
	$tab(index).table configure -rows [expr {[llength $ilist]+1}]

	set row 1
	foreach data $ilist {
		foreach {name unique cols} $data {
			if {$unique==1} {
				set unique UQ
			} else {
				set unique {}
			}
			set index($row,0) $name
			set index($row,1) $unique
			set index($row,2) [join $cols {, }]
			set index($row,3) $db
			set index($row,4) $table
		}
		incr row
	}
}


proc Property::setTrigger {db table} {
	variable trigger
	variable tab
	#element of table
	set colname 0
	
	set tlist [::Sqlite::getTriggerList $db $table]

	$tab(trigger).table selection clear all
	$tab(trigger).table configure -rows [expr {[llength $tlist]+1}]

	set row 1
	foreach data $tlist {
		foreach {name sql} $data {
			# regsub -all -- {\r|\n} $sql { } sql
			set trigger($row,0) $name
			set trigger($row,1) $sql
			set trigger($row,2) $db
		}
		incr row
	}
	set selected [lindex [$tab(trigger).table curselection] 0]
	$tab(trigger).table selection clear all
}

# One selection
# return [list db table index uniq cols sql]
proc Property::getSelectedIndex {} {
	variable tab
	variable index
	if {$::database(name) eq {}} {return}
	set table $tab(index).table

	set selection [lindex [$table curselection] 0]
	set row [$table index $selection row]
	if {$selection eq {} || $row == 0} {return}

	set iname $index($row,0)
	set uniq  $index($row,1)
	set cols  $index($row,2)
	set db    $index($row,3)
	set table $index($row,4)
	set master [Sqlite::getSqliteMasterName $db]
	set sql [Cmd::evalSQL "select sql from $master where type = 'index' and name = '$iname';"]
	set sql [lindex $sql 0]
	return [list $db $table $iname $uniq $cols $sql]
}

;#>>>

# this can create one more widgets
namespace eval SQLText {;#<<<
	variable info
	variable tag
	variable widgets

	array set info {
		highlight     1
		tab           4
	}
	set info(font) [list [font configure TkTextFont -family] [font configure TkTextFont -size]]
	set info(style,keyword) {fg RoyalBlue2 bg white bold 0 underline 0 italic 0 uppercase 0}
	set info(style,type)    {fg DarkOrange bg white bold 0 underline 0 italic 0 uppercase 0}
	set info(style,quote)   {fg chocolate3 bg white bold 0 underline 0 italic 0 uppercase 0}
	set info(style,comment) {fg green3     bg white bold 0 underline 0 italic 0 uppercase 0}

	array set tag {
		keyword tagKeyword
		type	tagType
		quote	tagQuote
		comment	tagComment
	}
	variable keywordList {
		alter table rename to add column
		analyze
		attach database as detach
		begin deferred immediate exclusive transaction end commit rollback
		copy or from using delimiters 
		create unique index on conflict
		temp temporary table constraint not null primary key autoincrement
		default collate check
		trigger before after instead of
		drop view
		escape isnull notnull between and or in exists case when then else cast like glob regexp
		insert into values abort fail ignore replace
		select all distinct where group by having order by limit offset
		natural left right full outer inner cross join on using collate asc desc
		pragma
		reindex
		replace
		union intersect except
		update set
		vacuum explain
	}
	set keywordList [lsort -uniq $keywordList]
	variable typeList {blob bit boolean bool char integer int numeric number decimal dec float real date time datetime timestamp interval text string current_timestamp current_date current_time}

	trace add variable ::SQLText::info(font)          write {SQLText::changeStyle}
	trace add variable ::SQLText::info(tab)           write {SQLText::changeStyle}
	trace add variable ::SQLText::info(style,keyword) write {SQLText::changeStyle}
	trace add variable ::SQLText::info(style,type)    write {SQLText::changeStyle}
	trace add variable ::SQLText::info(style,quote)   write {SQLText::changeStyle}
	trace add variable ::SQLText::info(style,comment) write {SQLText::changeStyle}
}

proc SQLText::new {path} {
	variable widgets
	
	text $path -relief flat -undo yes -highlightthickness 0 -height 0
	# Hook
	bind $path <KeyRelease> {
		::SQLText::highlight %W "insert linestart" "insert lineend"
	}
	bind $path <<Paste>> {
		tk_textPaste %W
		::SQLText::highlight %W "0.0" end
		break
	}
	bind $path <<Cut>> {
		tk_textCut %W
		::SQLText::highlight %W "0.0" end
		break
	}
	bind $path <Destroy> [namespace code {
		set widgets [::Util::lexcept $widgets %W]
	}]
	lappend widgets $path
	changeStyle
	return $path
}

proc SQLText::setTagStyle {path type font data} {
	variable tag
	foreach {key val} $data {
		switch -- $key {
			bold      { if {$val} {lappend font bold}      }
			underline { if {$val} {lappend font underline} }
			italic    { if {$val} {lappend font italic}    }
			default   { set $key $val }
		}
	}
	$path tag configure $tag($type) \
		-background $bg \
		-foreground $fg \
		-font $font
}

proc SQLText::updateTextStyle {} {
	variable info
	variable widgets

	set n [expr {72.0 / [winfo pixels . 1i]}]
	foreach path $widgets {
		$path configure \
			-font $info(font) \
			-tabs [expr {$n * [font measure $info(font) 0] * $info(tab)}]
		
		setTagStyle $path keyword $info(font) $info(style,keyword)
		setTagStyle $path type    $info(font) $info(style,type)
		setTagStyle $path quote   $info(font) $info(style,quote)
		setTagStyle $path comment $info(font) $info(style,comment)
		$path tag raise sel
	}
}

proc SQLText::changeStyle {args} {
	#TODO: refactor
	::SQLText::updateTextStyle
	::Tool::DataEditor::updateTextStyle
}

proc SQLText::highlight {path stIndex edIndex} {
	variable info
	variable tag
	variable keywordList
	variable typeList
	if {$info(highlight) == 0} return
	
	# Comment
	$path tag remove $tag(comment) $stIndex $edIndex
	set ed 0.0
	while 1 {
		set st [$path search -exact -- "/*" $ed end] 
		if {$st eq ""} {break}
		set ed [$path search -exact -- "*/" "$st +1 c" end] 
		if {$ed eq ""} {break}
		$path tag add $tag(comment) $st "$ed + 2 c"
	}
	set ed 0.0
	while 1 {
		set st [$path search -exact -- "--" $ed end] 
		if {$st eq ""} {break}
		set ed [lindex [split $st .] 0].end
		$path tag add $tag(comment) $st $ed
	}

	# Quote
	$path tag remove $tag(quote) $stIndex $edIndex
	foreach {lq rq} {\" \" ' ' \[ \]} {
		set ed 0.0
		while 1 {
			set st [$path search -exact -- $lq $ed end] 
			if {$st eq {}} {break}
			set ed [$path search -exact -- $rq "$st +1 c" end] 
			if {$ed eq {}} {break}
			$path tag add $tag(quote) $st "$ed + 1 c"
			set ed "$ed + 1 c"
		}
	}
	
	# Keywords
	# word list for lintersect. for speed up
	set wlist [string tolower [string map {; " " , " " . " " ( " " ) " " \" " " ' " " \{ " " \} " "} \
			[$path get $stIndex $edIndex]]]
	set wlist [lsort -uniq $wlist]

	foreach tagName {keyword type} wordList [list $keywordList $typeList] {
		$path tag remove $tag($tagName) $stIndex $edIndex
		set newkeys [::Util::lintersect $wlist $wordList]
		foreach k $newkeys {
			set index $stIndex
			while 1 {
				if {[$path compare $index >= end]} break
				set len 0
				set index [$path search -nocase -count len -- $k $index $edIndex] 
				if {$len == 0} break
				set c ""
				append c [$path get "$index - 1 c" $index]
				append c [$path get "$index + $len c" "$index + $len c + 1 c"]
				set c [string map {"(" "" ")" "" "," "" "." "" " " "" ";" "" "\t" "" "\n" ""} $c]
				if {$c ne ""} {
					set index "$index + $len c"
					continue
				}
				$path tag add $tag($tagName) $index "$index + $len c"
				set index "$index + $len c + 1 c"
			}
		}
	}
	# Uppercase Keywords 
	foreach type {keyword type quote comment} {
		if {![dict get $info(style,$type) uppercase]} continue
		set tagName $tag($type)
		set index $stIndex
		while 1 {
			set range [$path tag nextrange $tagName $index $edIndex]
			if {[llength $range] == 0} break
			foreach {st ed} $range break
			set rtext [$path get $st $ed]
			set utext [string toupper $rtext]
			if {$rtext ne $utext} {
				set cursor [$path index insert]
				$path insert $ed $utext $tagName
				$path delete $st $ed
				$path mark set insert $cursor
			}
			set index "$ed + 1 c"
		}
	}
	return
}

proc SQLText::dump {} {
	variable info
	set session "#::[info level [info level]]\n"
	::append session "array set [namespace current]::info \[list [array get info]\]\n"
	return $session
}
;#>>>

namespace eval Editor {;#<<<
}

# setup widgets
proc Editor::Editor {} {
	variable info
	variable tag
	ttk::frame .editor -height 50

	# Title & toolbuttons
	set title [Titlebar::Titlebar .editor.title [msgcat::mc "SQL Command"] {}]

	# widgets
	ttk::frame .editor.f -style TEntry -border 2
	grid $title -row 0 -column 0 -sticky ew
	grid .editor.f -row 1 -column 0 -sticky news
	grid columnconfigure .editor 0 -weight 1
	grid rowconfigure    .editor 1 -weight 1
	
	ttk::scrollbar .editor.f.y -command {.editor.f.text yview}
	::SQLText::new .editor.f.text
	.editor.f.text configure -yscrollcommand {.editor.f.y set}
	grid .editor.f.text -row 0 -column 0 -sticky news
	grid .editor.f.y    -row 0 -column 1 -sticky ns
	grid columnconfigure .editor.f 0 -weight 1
	grid rowconfigure    .editor.f 0 -weight 1
	
	set m [menu .editor.pop]
	$m add command -label [msgcat::mc Undo] -accelerator "$::pref(shortmod)+Z" -command { catch {.editor.f.text edit undo} {} }
	$m add command -label [msgcat::mc Redo] -accelerator "$::pref(shortmod)+Y" -command  { catch {.editor.f.text edit redo} {} }
	$m add separator
	$m add command -label [msgcat::mc Cut] -accelerator "$::pref(shortmod)+X" \
		-command {event generate .editor.f.text <<Cut>>}
	$m add command -label [msgcat::mc Copy] -accelerator "$::pref(shortmod)+C" \
		-command {event generate .editor.f.text <<Copy>>}
	$m add command -label [msgcat::mc Paste] -accelerator "$::pref(shortmod)+V" \
		-command {event generate .editor.f.text <<Paste>>}
	$m add command -label [msgcat::mc Delete] -accelerator "$::pref(shortmod)-D" \
		-command "event generate .editor.f.text <$::pref(modifier)-d>"
	$m add separator
	$m add command -label [msgcat::mc "Select All"] -accelerator "$::pref(shortmod)+A" \
		-command "event generate .editor.f.text <$::pref(modifier)-a>"
	$m add command -label [msgcat::mc "Delete All"] -accelerator "$::pref(shortmod)+Del" \
		-command "event generate .editor.f.text <$::pref(modifier)-Delete>"
	$m add separator
	$m add command -label [msgcat::mc "Run SQL"] -accelerator "$::pref(shortmod)+Return" \
		-command "event generate .editor.f.text <$::pref(modifier)-Return>"

	# binding
	bind .editor.f.text <<B3>> {focus -force %W; tk_popup .editor.pop %X %Y; break;}
	bind .editor.f.text <$::pref(modifier)-Return>    {::Cmd::executeSqlCmd; break;}
	bind .editor.f.text <$::pref(modifier)-Delete>    {::Editor::clear}
	bind .editor.f.text <$::pref(modifier)-BackSpace> {::Editor::clear}
	Cmd::bindTextFocusTtkEntry .editor.f.text .editor.f
	return .editor
}

proc Editor::getText {} {
	if {[llength [.editor.f.text tag ranges sel]] > 0} {
		return [.editor.f.text get sel.first sel.last]
	} else {
		return [.editor.f.text get 1.0 "end-1 char"]
	}
}

proc Editor::setText {str} {
	.editor.f.text delete 1.0 end
	.editor.f.text insert 1.0 [string map {\r {}} $str]
	::SQLText::highlight .editor.f.text 0.0 end
}

proc Editor::clear {} {
	.editor.f.text delete 1.0 end
}


proc Editor::focusThis {} {
	focus -force .editor.f.text
}

;#>>>


namespace eval Table {;#<<<
	variable title {}
	array set table {}
	array set rawdata {};#this is for sqlite3 and not utf-8 encoding
	array set rowid {}
	
	variable editable 0
	array set edit {
		start 0
		index {}
		colnames {}
		db {}
		table {}
		db.table {}
	}
	# Table widget
	variable tablew
	
	# info
	# vibind     : vi like cursor move
	variable info
	array set info {
		vibind		0
	}

	# For double key push handling
	namespace eval key {
		variable doubleD {}
		variable pretime 0
		variable pushedD  0
	}
}

proc Table::dump {} {
	variable info
	set session "#::[info level [info level]]\n"
	::append session "array set [namespace current]::info \[list [array get info]\]\n"
	return $session
}

namespace eval Table::CellEditor {;#<<<
	variable ttable
	variable parent
	variable root
	variable isPop 0
	array set info {
		start 0
		index ""
		rowid ""
		colnames {}
	}
}
#this widget will be used for editing a cell.
proc Table::CellEditor::CellEditor {parentFrame Table} {
	variable ttable $Table
	variable parent $parentFrame
	variable root
	set root $parent.celleditor
	frame $root -bd 2 -bg black
	text $root.text -border 0 -padx 0 -pady 0 \
		-relief flat -undo 1 -highlightthickness 0
	pack $root.text -expand yes -fill both

	if {[tk windowingsystem] eq "x11"} {
		bind $root.text <FocusOut> {
			if {$::Table::CellEditor::isPop == 0} {
				Table::CellEditor::end
			}
		}
		bind $root.text <FocusIn> {
			if {$::Table::CellEditor::isPop == 1} {
				set Table::CellEditor::isPop 0
				focus -force %W
			}
		}
		bind $root.text <ButtonRelease-3> {
			set Table::CellEditor::isPop 1
			tk_popup [winfo parent %W].pop %X %Y
			break
		}
	} else {
		bind $root.text <FocusOut> { Table::CellEditor::end }
		bind $root.text <<B3>> \
			{ tk_popup [winfo parent %W].pop %X %Y; break }
	}
	bind $root.text <Escape> {
		Table::CellEditor::end 1
		::Table::Navi .table.f.table center
	}
	bind $root.text <Return> {
		Table::CellEditor::end
		::Table::Navi .table.f.table down
		break
	}
	bind $root.text <Shift-Return> {
		Table::CellEditor::end
		::Table::Navi .table.f.table up
		break
	}
	bind $root.text <$::pref(modifier)-Return> {
		Table::CellEditor::end
		::Table::Navi .table.f.table center
		break
	}
	bind $root.text <Alt-Return> {
		%W insert insert \n
		break
	}
	bind $root.text <Key-Up> {
		if {[lindex [split [%W index insert] .] 0] == 1} {
		Table::CellEditor::end
		::Table::Navi .table.f.table up
		}
	}
	bind $root.text <Key-Down> {
		if {[lindex [split [%W index insert] .] 0] == 
		[lindex [split [%W index {end - 1 chars}] .] 0]} {
		Table::CellEditor::end
		::Table::Navi .table.f.table down
		}
	}
	bind $root.text <Key-Left> {
		if {[%W index insert] eq [%W index {1.0}]} {
		Table::CellEditor::end
		::Table::Navi .table.f.table left
		}
	}
	bind $root.text <Key-Right> {
		if {[%W index insert] eq [%W index {end - 1 chars}]} {
		Table::CellEditor::end
		::Table::Navi .table.f.table right
		}
	}
	bind $root.text <Key-Tab> {
		event generate .table.f.table <$::pref(modifier)-Return>
		update
		set _col [.table.f.table index [.table.f.table tag cell active] col]
		if {[incr _col] >= [.table.f.table cget -cols]} {
			::Table::Navi .table.f.table down
			::Table::Navi .table.f.table head
		} else {
			::Table::Navi .table.f.table right
		}
		event generate .table.f.table <Key-Return>
		break
	}
	bind $root.text <Shift-Tab> {
		event generate .table.f.table <$::pref(modifier)-Return>
		update
		set _col [.table.f.table index [.table.f.table tag cell active] col]
		if {$_col <= 1} {
			::Table::Navi .table.f.table up
			::Table::Navi .table.f.table end
		} else {
			::Table::Navi .table.f.table left
		}
		event generate .table.f.table <Key-Return>
		break
	}
	trace add variable ::Table::info(vibind) write Table::CellEditor::useViBind
	
	# CellEditor pop
	Util::makeEditorPop $root
	
	return $root
}


proc Table::CellEditor::useViBind {n1 n2 op} {
	variable ttable
	variable root
	set text $root.text
	# n1(n2) = Table::info(vibind) maybe
	if [set ${n1}($n2)] {
		bind $text <Control-h> \
		"event generate %W <Control-Return>
		Table::Navi $ttable left; update
		event generate %W <Return>; break"

		bind $text <Control-j> \
		"event generate %W <Control-Return>
		Table::Navi $ttable down; update
		event generate %W <Return>; break"

		bind $text <Control-k> \
		"event generate %W <Control-Return>
		Table::Navi $ttable up; update
		event generate %W <Return>; break"

		bind $text <Control-l> \
		"event generate %W <Control-Return>
		Table::Navi $ttable right; update
		event generate %W <Return>; break"
	} else {
		bind $text <Control-h> {}
		bind $text <Control-j> {}
		bind $text <Control-k> {}
		bind $text <Control-l> {}
	}
}

proc Table::CellEditor::begin {index tablename colnames} {
	variable ttable
	variable parent
	variable root
	variable info
	if {$info(start)!=0} return
	if {[scan $index %d,%d row col] != 2} return
	dputs {Table::CellEditor::begin : row = $row, col = $col}
	if {$row==0 || $col==0} return

	dputs {Table::CellEditor::begin  : start}
	
	set info(table) $tablename
	set info(start) 1
	set info(index) $index
	set info(rowid) $::Table::rowid($row,1)
	set info(colnames) $colnames
	$root.text delete 1.0 end
	$root.text insert 1.0 [$ttable get active]
	$root.text edit reset
	$root.text tag add sel 1.0 {end -1 char}
	
	$ttable window configure $index -window $root -sticky news	
	focus -force $root.text
}

proc Table::CellEditor::end {{escape 0}} {
	variable ttable
	variable parent
	variable root
	variable info

	if {$info(start) != 1} return
	set info(start) 0
	dputs {Table::CellEditor::end  : start}

	set oldval [$ttable get active]
	set newval [$root.text get 1.0 {end-1 char}]
	
	# Change
	if {($escape == 0) && ($oldval ne $newval)} {
		set table $info(table);
		set rowid $info(rowid)
		scan $info(index) {%d,%d} row col
		
		# TODO : make option selecting null string is NULL or ''.
		set value [Util::quoteValue $newval "NULL"]

		# Update
		set colname [lindex $info(colnames) [expr {$col-1}]]
		set query "UPDATE $table SET \"$colname\" = $value WHERE rowid = $rowid;"
		set oldcount $::Sqlite::total_changes
		if {[::Sqlite::evalQuery $query] == 0} {
			set count [expr {$::Sqlite::total_changes - $oldcount}]
			set ::Table::table($info(index)) $newval

			if {[Sqlite::getCurrentVersion] > 2 && 
				$::database(encoding) ne "utf-8"} {
				::Cmd::encodingConvert to newval
				set ::Table::rawdata($info(index)) $newval 
			}
			
			# if this row was integer primary key, the column is
			# same as rowid. We must sync Table::rowid data.
			set query "SELECT rowid FROM $table WHERE rowid = $rowid;"
			if {[::Sqlite::evalQuery $query] == 1} {#error}
			if {$::Sqlite::data eq {}} {
				set ::Table::rowid($row,1) $value
			}
			
;#            if {[Sqlite::getCurrentVersion] == 3 && $count > 1} {
;#                ::Message::show -type apperror -textbox 0 \
;#                -message [msgcat::mc "More than one rows were updated by trigger.  Data of table may not be synchronized.  Please re-open this table for synchronizing."]
;#            }
			::update
			[namespace parent]::updateDataEditor
		}
	}
	$ttable window configure $info(index) -window {}
	focus -force $ttable
}
;#>>>

namespace eval Table::RowEditor {;#<<<
	variable ttable
	variable parent
	variable root
	variable isPop 0
	array set info {
		start 0
		index ""
		rowid ""
		colnames {}
	}
}

proc Table::RowEditor::RowEditor {parentFrame Table} {;#<<<
	variable ttable $Table
	variable parent $parentFrame
	variable root
	set root $parent.roweditor
	frame $root -bd 2 -bg black
	text $root.text -border 0 -takefocus 1 -padx 0 -pady 0 \
		-relief flat -undo 1 -highlightthickness 0
	pack $root.text -expand yes -fill both

	if {[tk windowingsystem] eq "x11"} {
		bind $root.text <FocusOut> {
			if {$::Table::RowEditor::isPop == 0} {
				Table::RowEditor::end 1
			}
		}
		bind $root.text <FocusIn> {
			if {$::Table::RowEditor::isPop == 1} {
				set Table::RowEditor::isPop 0
				focus -force %W
			}
		}
		bind $root.text <ButtonRelease-3> {
			set Table::RowEditor::isPop 1
			tk_popup [winfo parent %W].pop %X %Y
			break
		}
	} else {
		bind $root.text <FocusOut> {Table::RowEditor::end 1}
		bind $root.text <<B3>> \
			{ tk_popup [winfo parent %W].pop %X %Y; break}
	}
	bind $root.text <Escape> {
		Table::RowEditor::end 1
		::Table::Navi .table.f.table center
	}
	bind $root.text <Return> {
		Table::RowEditor::end
		::Table::Navi .table.f.table center
		break
	}
	bind $root.text <Shift-Return> {
		Table::CellEditor::end
		::Table::Navi .table.f.table up
		break
	}
	bind $root.text <$::pref(modifier)-Return> {
		Table::RowEditor::end
		::Table::Navi .table.f.table center
		break
	}
	bind $root.text <Alt-Return> {
		%W insert insert \n
		break
	}
	bind $root.text <Key-Up> {
		Table::RowEditor::move up
		break
	}
	bind $root.text <Key-Down> {
		Table::RowEditor::move down
		break
	}
	bind $root.text <Key-Left> {
		if {[%W index insert] eq [%W index {1.0}]} {
		Table::RowEditor::move left
		break
		}
	}
	bind $root.text <Key-Right> {
		if {[%W index insert] eq [%W index {end - 1 chars}]} {
		Table::RowEditor::move right
		break
		}
	}
	bind $root.text <Key-Tab> {
		Table::RowEditor::move right
		break
	}
	bind $root.text <Shift-Tab> {
		Table::RowEditor::move left
		break
	}
	trace add variable ::Table::info(vibind) write Table::RowEditor::useViBind
	
	# CellEditor pop
	Util::makeEditorPop $root
	
	return $root
};#>>>


proc Table::RowEditor::useViBind {n1 n2 op} {
	variable ttable
	variable root
	set text $root.text
	# n1(n2) = Table::info(vibind) maybe
	if [set ${n1}($n2)] {
		bind $text <Control-h> { event generate %W <Key-Left>; break }
		bind $text <Control-j> { event generate %W <Key-Down>; break }
		bind $text <Control-k> { event generate %W <Key-Up>; break }
		bind $text <Control-l> { event generate %W <Key-Right>; break }
	} else {
		bind $text <Control-h> {}
		bind $text <Control-j> {}
		bind $text <Control-k> {}
		bind $text <Control-l> {}
	}
}


# moving between cells left <-> right
proc Table::RowEditor::move {dir {count 1}} {
	variable ttable
	variable root
	variable info
	if {$info(start) != 1} {return}

	set row $info(row)
	set col $info(col)
	$ttable configure -state normal
	$ttable set $row,$col [$root.text get 1.0 {end-1 char}]
	$ttable configure -state disabled
	
	set colmax [llength $info(colnames)]
	
	if {$dir eq "absolute"} {set col $count}
	if {$dir eq "up" || $dir eq "down"} {
		if {[Table::RowEditor::end] == 0} {
			switch -exact -- $dir {
				up   { set index [expr {$row-1}],$col }
				down { set index $row,$col }
			}
			$ttable selection clear all
			$ttable selection set $index $index
			$ttable activate $index
			focus -force $ttable
			return
		}
		return
	}
	if {$dir eq "left"}  { set col [expr {$col - $count}] }
	if {$dir eq "right"} { set col [expr {$col + $count}] }
	if {$col < 1} {set col $colmax}
	if {$col > $colmax} {set col 1}
	
	set info(row) $row
	set info(col) $col
	
	$root.text delete 1.0 end
	$root.text insert 1.0 [$ttable get $row,$col]
	$root.text edit reset
	$root.text tag add sel 1.0 {end -1 char}

	$ttable selection clear all
	$ttable selection set $row,1 $row,$colmax
	$ttable activate $row,$col
	$ttable window configure $row,$col -window $root -sticky news	
	focus -force $root.text
}

proc Table::RowEditor::begin {db table colnames autoIncrIndex} {
	dputs {Table::RowEditor::begin : db=$db table=$table
	colnames=$colnames autoIncrIndex=$autoIncrIndex}
	variable ttable
	variable root
	variable info
	if {$info(start)!=0} return
	
	set colmax [llength $colnames]
	set row [$ttable cget -rows]

	set col 1
	set info(table) "'$db'.'$table'"
	set info(start) 1
	set info(row)   $row
	set info(col)	$col
	set info(colnames) $colnames
	set info(col.autoincr) $autoIncrIndex
	$root.text delete 1.0 end
	$root.text insert 1.0 ""
	$root.text edit reset

	set columninfolist [::Sqlite::getColumnList $db $table]
	set sqliteversion [::Sqlite::getCurrentFullVersion]

	$ttable configure -state normal
	$ttable configure -rows [expr {$row+1}]
	for {set col 1} {$col <= $colmax} {incr col} {
		set columninfo [lindex $columninfolist [expr {$col-1}]]

		# set default value if column has default.
		if {[dict get $columninfo defisnull]} {
			$ttable set $row,$col ""
		} else {
			set defaultvalue [dict get $columninfo def]
			if {[::Util::versionCompare $sqliteversion "3.3.7"] < 1} {
				set columntype [dict get $columninfo type]
				set defaultvalue [::Util::getDefaultValue $sqliteversion $columntype $defaultvalue]
			}
			set query "SELECT $defaultvalue"
			::Sqlite::evalQuery $query {time 0}
			$ttable set $row,$col [lindex [set ::Sqlite::data] 0]
		}
		dputs {$row,$col}
	}
	$ttable configure -state disabled

	$root.text delete 1.0 end
	$root.text insert 1.0 [$ttable get $row,1]
	$root.text edit reset
	$root.text tag add sel 1.0 {end -1 char}

	$ttable selection clear all
	$ttable tag row newcol $row
	$ttable activate $row,1
	$ttable window configure $row,1 -window $root -sticky news	
	$ttable see $row,1
	focus -force $root.text
}

proc Table::RowEditor::end {{escape 0}} {
	dputs {Table::RowEditor::end escape=$escape}
	variable ttable
	variable parent
	variable root
	variable info
	set row $info(row)
	set col $info(col)
	if {$info(start) != 1} { return 0 }
	set info(start) 0

	# Cancel
	if {$escape == 1} {
		$ttable window configure $row,$col -window {}
		set trow [expr {[$ttable cget -rows] -1}]
		$ttable configure -rows $trow
		focus -force $ttable
		[namespace parent]::updateDataEditor
		$ttable tag row default $row
		return 0
	}
	$ttable configure -state normal
	$ttable set $row,$col [$root.text get 1.0 {end-1 char}]
	$ttable configure -state disabled

	# Try insert new row
	set version [Sqlite::getCurrentVersion]
	set table $info(table);
	
	set colnames [list]
	foreach {colname} $info(colnames) {
		lappend colnames '$colname'
	}
	set query "INSERT INTO $table ([join $colnames ,]) VALUES "
	set colvals [list]
	set colmax [llength $info(colnames)]
	for {set col 1} {$col <= $colmax} {incr col} {
		set val [Util::quoteValue [$ttable get $row,$col] "NULL"]
		lappend colvals $val
	}

	# if an auto increment column is empty, then it may be NULL.
	set autoincr $info(col.autoincr)
	if {$autoincr ne {}} {
		if {[lindex $colvals $autoincr] eq "''"} {
			lset colvals $autoincr NULL
		}
	}

	append query "([join $colvals ,]);"

	set oldcount $::Sqlite::total_changes
	if {[::Sqlite::evalQuery $query {time 1}] != 0} {
		dputs {failed to insert row. I gave up end this mode.}
		set info(start) 1
		return 1
	}
	set count [expr {$::Sqlite::total_changes - $oldcount}]

	set rowid [::Sqlite::getLastInsertRowid]
	set query "SELECT * FROM $table WHERE rowid = $rowid"
	Cmd::encodingConvert to query
	if {[::Sqlite::evalQuery $query {time 1 encode 0}] != 0} {}
	
	if {[Sqlite::getCurrentVersion] > 2 && 
		$::database(encoding) ne "utf-8"} {
		set col 1
		foreach rd $::Sqlite::data {
			set d $rd
			Cmd::encodingConvert from d
			set ::Table::table($row,$col) $d
			set ::Table::rawdata($row,$col) $rd
			incr col
		}
	} else {
		set col 1
		foreach d $::Sqlite::data {
			Cmd::encodingConvert from d
			set ::Table::table($row,$col) $d
			incr col
		}
	}
	set ::Table::rowid($row,1) $rowid
	
;#    if {[Sqlite::getCurrentVersion] == 3 && $count > 1} {
;#        ::Message::show -type apperror -textbox 0 \
;#        -message [msgcat::mc "More than one rows were updated by trigger.  Data of table may not be synchronized.  Please re-open this table for synchronizing."]
;#    }
	
	$ttable window configure $row,$info(col) -window {}
	$ttable selection clear all
	$ttable selection set active
	$ttable tag row default $row
	focus -force $ttable
	[namespace parent]::updateDataEditor
	return 0
}
;#>>>

proc Table::Table {} {;#<<<
	variable info
	variable tablew
	variable title

	ttk::frame .table
	set title [Titlebar::Titlebar .table.title [msgcat::mc Table] {}]
	
	#
	# Table
	#
	ttk::frame .table.f -style TEntry -border 2
	grid $title   -row 0 -column 0 -sticky ew
	grid .table.f -row 1 -column 0 -sticky news
	grid columnconfigure .table 0 -weight 1
	grid rowconfigure    .table 1 -weight 1
	
	ttk::scrollbar .table.f.x -command {.table.f.table xview} -orient horizontal
	ttk::scrollbar .table.f.y -command {.table.f.table yview}
	table .table.f.table \
		-rows 0 -cols 0 \
		-variable ::Table::table -cache 0 \
		-selectmode extended \
		-titlerows 1 -titlecols 1 -selecttitle 0 \
		-xscrollcommand {.table.f.x set} \
		-yscrollcommand {.table.f.y set}
	.table.f.table width 0 3
	set tablew .table.f.table
	::Cmd::changeTableLookAndFeel $tablew
	ttk::frame .table.f.bg
	grid .table.f.x -row 1 -column 0 -sticky ew
	grid .table.f.y -row 0 -column 1 -sticky ns
	grid $tablew    -row 0 -column 0 -sticky news
	grid .table.f.bg -row 1 -column 1 -sticky news
	grid columnconfigure .table.f 0 -weight 1
	grid rowconfigure    .table.f 0 -weight 1
	grid propagate .table 0

	# update DataEditor
	$tablew configure -browsecmd [namespace code {
		set _row [%W index active row]
		set _col [%W index active col]
		if {$_row < 1} {set _row 1}
		if {$_col < 1} {set _col 1}

		%W activate $_row,$_col
		
		unset -nocomplain -- _cell _row _col
		updateDataEditor
	}]

	
	# Table Pop Menu
	menu .table.pop
	.table.pop add command -label [msgcat::mc Copy] -accelerator "$::pref(shortmod)+C" \
		-command "tk_tableCopy $tablew"
	.table.pop add separator
	.table.pop add command -label [msgcat::mc "Sort Asc as String"] -command {
		::Table::sort [.table.f.table index active col] incr dictionary
	}
	.table.pop add command -label [msgcat::mc "Sort Desc as String"] -command {
		::Table::sort [.table.f.table index active col] decr dictionary
	}
	.table.pop add separator
	.table.pop add command -label [msgcat::mc "Sort Asc as Number"] -command {
		::Table::sort [.table.f.table index active col] incr number
	}
	.table.pop add command -label [msgcat::mc "Sort Desc as Number"] -command {
		::Table::sort [.table.f.table index active col] decr number
	}
	.table.pop add separator
	.table.pop add command -label [msgcat::mc Search]... -accelerator F3 -command {
		::Tool::Searcher::run
	}
	.table.pop add command -label [msgcat::mc "Data Editor"]... -accelerator F4 -command {
		::Tool::DataEditor::run
	}
	.table.pop add separator
	.table.pop add command -label [msgcat::mc Update] -accelerator F5 -command {
		::Table::update
	}
	.table.pop add separator
	.table.pop add command -label [msgcat::mc "Reset Row Size"] -command {
		foreach _sel [.table.f.table tag row sel] {
			.table.f.table height $_sel default
		}
	}
	.table.pop add command -label [msgcat::mc "Reset Column Size"] -command {
		foreach _sel [.table.f.table tag col sel] {
			.table.f.table width $_sel default
		}
	}
	.table.pop add separator
	.table.pop add command -label [msgcat::mc "Insert New Row"] -accelerator "$::pref(shortmod)+N" \
		-state disabled -command "event generate $tablew <$::pref(modifier)-n>"
	.table.pop add command -label [msgcat::mc "Delete Row"] -accelerator "$::pref(shortmod)+D" \
		-state disabled -command "event generate $tablew <$::pref(modifier)-d>"

	# binding
	bind $tablew <$::pref(modifier)-n> {::Table::insertRow}
	bind $tablew <$::pref(modifier)-d> {::Table::deleteRow}
	trace add variable ::Table::info(vibind) write Table::useViBind

	Cmd::bindTextFocusTtkEntry $tablew .table.f

	bind $tablew <1> {
		dputs {Button1=[%W index @%x,%y]}
		if {$::Table::CellEditor::info(start) == 1} {
			dputs {Now Editing a Cell...?}
			::Table::CellEditor::end
		} elseif {$::Table::RowEditor::info(start) == 1} {
			dputs {Now Creating New Row...?}
			if {[%W index @%x,%y row]+1 != [%W cget -rows]} {
				dputs {[%W index @%x,%y row]+1 != [%W cget -rows]}
				dputs {Dismiss}
				if {[::Table::RowEditor::end] == 1} {
					::Table::RowEditor::move center 0
					break
				}
			} else {
				dputs {Move}
				::Table::RowEditor::move absolute [%W index @%x,%y col]
				break
			}
		} else {
			dputs {Now Normal Mode}
			focus -force %W
			if {[%W index @%x,%y row] > 0 && [%W index @%x,%y col] > 0} {
				%W activate [%W index @%x,%y]
			}
		}
	}
	bind $tablew <<B3>> {
		dputs {Button3=[%W index @%x,%y]}
		if {$::Table::CellEditor::info(start) == 1} {
			::Table::CellEditor::end
		} elseif {$::Table::RowEditor::info(start) == 1} {
			dputs {Now Creating New Row...?}
			if {[%W index @%x,%y row]+1 != [%W cget -rows]} {
				dputs {[%W index @%x,%y row]+1 != [%W cget -rows]}
				dputs {Dismiss}
				if {[::Table::RowEditor::end] == 1} {
					::Table::RowEditor::move center 0
					break
				}
			} else {
				dputs {Move}
				::Table::RowEditor::move absolute [%W index @%x,%y col]
				break
			}
		} else {
			dputs {Now Normal Mode}
		}
		focus -force %W
		if { ![%W selection includes [%W index @%x,%y]] } {
			## select one cell
			%W activate [%W index @%x,%y]
			if {[%W index @%x,%y row] > 0 && [%W index @%x,%y col] > 0} {
				::Table::Navi %W center
			}
		}
		tk_popup .table.pop %X %Y
		break
	}

	bind $tablew <Tab> {break};# do nothing
	bind $tablew <F5> {::Table::update}

	#Debug Print
	deval { bind $tablew <KeyRelease> {
		puts "table active = [.table.f.table tag cell active]"
	} }

	# Create EditMode Object
	CellEditor::CellEditor .table $tablew
	RowEditor::RowEditor .table $tablew
	
	return .table
}

proc Table::useViBind {n1 n2 op} {
	variable tablew
	# n1(n2) = Table::info(vibind) maybe
	if [set ${n1}($n2)] {
		bind $tablew <h> "Table::Navi $tablew left"
		bind $tablew <j> "Table::Navi $tablew down"
		bind $tablew <k> "Table::Navi $tablew up"
		bind $tablew <l> "Table::Navi $tablew right"
		bind $tablew <o> "Table::insertRow; break"
		bind $tablew <slash>    "GUICmd::Search"
		bind $tablew <Double-g> "Table::Navi $tablew top"
		bind $tablew <G>        "Table::Navi $tablew bottom"
		bind $tablew <u>		"event generate $tablew <F5>; break"
		set key::pretime 0
		set key::pushedD 0
		bind $tablew <d> [namespace code {
			set key::now [clock clicks -milliseconds]
			if {$key::pushedD == 1 && [expr {$key::now-$key::pretime}] < 500} {
				deleteRow
				set key::pushedD 0
			} else {
				set key::pushedD 1
			}
			set key::pretime $key::now
			break
		}]
	} else {
		bind $tablew <h> {}
	Table::focusThis
		bind $tablew <j> {}
		bind $tablew <k> {}
		bind $tablew <l> {}
		bind $tablew <o> {}
		bind $tablew <slash>    {}
		bind $tablew <Double-g> {}
		bind $tablew <G>        {}
		bind $tablew <u>		{}
		bind $tablew <Double-d> {}
	}
}

proc Table::focusThis {} {
	variable tablew
	focus -force $tablew
	Table::see 1,1
}

proc Table::Navi {w dir} {
	switch -exact -- $dir {
		up    { set r -1; set c  0 }
		down  { set r  1; set c  0 }
		left  { set r  0; set c -1 }
		right { set r  0; set c  1 }
		center { set r  0; set c  0 }
		top    { set r 0; set c 0 ;#dummy }
		bottom { set r 0; set c 0 ;#dummy }
		head   { set r 0; set c 0 ;#dummy }
		end    { set r 0; set c 0 ;#dummy }
		default {return}
	}
	set index [$w tag cell active]
	if {$index eq {}} return
	set row [$w index active row]
	set col [$w index active col]
	incr row $r
	incr col $c

	set maxcol [$w cget -cols]
	set maxrow [$w cget -rows]

	switch -exact -- $dir {
		top    {set row 1}
		bottom {set row $maxrow}
		head   {set col 1}
		end    {set col $maxcol}
	}
	
	if {$row < 1} {incr row}
	if {$col < 1} {incr col}
	if {$row >= $maxrow} {set row $maxrow}
	if {$col >= $maxcol} {set col $maxcol}
	see $row,$col
}

# Sorting
proc Table::sort {colindex order {type dictionary}} {;#<<<
	variable tablew
	set rmax [$tablew cget -rows]
	set cmax [$tablew cget -cols]

	switch -exact -- $order {
		incr {set option "-increasing"} 
		decr {set option "-decreasing"}
		default {return}
	}
	switch -exact -- $type {
		number     {append option " -real"}
		dictionary {append option " -dictionary"}
	}

	if {[Sqlite::getCurrentVersion] > 2 && 
		$::database(encoding) ne "utf-8"} {
		sortWithRawData $colindex $option $rmax $cmax
	} else {
		sortWithoutRawData $colindex $option $rmax $cmax
	}
	::update
	updateDataEditor
};#>>>
proc Table::sortWithRawData {colindex option rmax cmax} {;#<<<
	variable editable
	variable table
	variable rowid
	variable rawdata

	if {$editable} {
		# sort rowid and data
		set st [clock clicks -milliseconds]
		set dataList [list]
		for {set r 1} {$r < $rmax} {incr r} {
			set dlist [list]
			lappend dlist $rowid($r,1)
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $table($r,$c)
			}
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $rawdata($r,$c)
			}
			lappend dataList $dlist
		}
		
		set r 1
		set dataList [eval lsort $option -index $colindex \$dataList]
		foreach dlist $dataList {
			set rowid($r,1) [lindex $dlist 0]
			set c 1
			foreach cdata [lrange $dlist 1 [expr {$cmax-1}]] {
				set table($r,$c) $cdata
				incr c
			}
			set c 1
			foreach cdata [lrange $dlist $cmax end] {
				set rawdata($r,$c) $cdata
				incr c
			}
			incr r
		}
		set ed [clock clicks -milliseconds]
		dputs {sort time : [expr $ed - $st]msec}
	} else {
		# sort data
		set st [clock clicks -milliseconds]
		incr colindex -1
		set dataList [list]
		for {set r 1} {$r < $rmax} {incr r} {
			set dlist [list]
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $table($r,$c)
			}
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $rawdata($r,$c)
			}
			lappend dataList $dlist
		}
		set r 1
		set sortedList [list]
		set sortedListRaw [list]
		set dataList [eval lsort $option -index $colindex \$dataList]
		foreach dlist $dataList {
			set c 1
			foreach cdata [lrange $dlist 0 $cmax] {
				lappend sortedList $r,$c $cdata
				incr c
			}
			set c 1
			foreach cdata [lrange [expr {$cmax +1}] end] {
				lappend sortedListRaw $r,$c $cdata
				incr c
			}
			incr r
		}
		array set table $sortedList
		array set rawdata $sortedListRaw
		set ed [clock clicks -milliseconds]
		dputs {sort time : [expr $ed - $st]msec}
	}
};#>>>
proc Table::sortWithoutRawData {colindex option rmax cmax} {;#<<<
	variable editable
	variable table
	variable rowid

	if {$editable} {
		# sort rowid and data
		set st [clock clicks -milliseconds]
		set dataList [list]
		for {set r 1} {$r < $rmax} {incr r} {
			set dlist [list]
			lappend dlist $rowid($r,1)
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $table($r,$c)
			}
			lappend dataList $dlist
		}
		
		set r 1
		set dataList [eval lsort $option -index $colindex \$dataList]
		foreach dlist $dataList {
			set rowid($r,1) [lindex $dlist 0]
			set c 1
			foreach cdata [lrange $dlist 1 end] {
				set table($r,$c) $cdata
				incr c
			}
			incr r
		}
		set ed [clock clicks -milliseconds]
		dputs {sort time : [expr $ed - $st]msec}
	} else {
		# sort data
		set st [clock clicks -milliseconds]
		incr colindex -1
		set dataList [list]
		for {set r 1} {$r < $rmax} {incr r} {
			set dlist [list]
			for {set c 1} {$c <$cmax} {incr c} {
				lappend dlist $table($r,$c)
			}
			lappend dataList $dlist
		}
		set r 1
		set sortedList [list]
		set dataList [eval lsort $option -index $colindex \$dataList]
		foreach dlist $dataList {
			set c 1
			foreach cdata $dlist {
				lappend sortedList $r,$c $cdata
				incr c
			}
			incr r
		}
		array set table $sortedList
		set ed [clock clicks -milliseconds]
		dputs {sort time : [expr $ed - $st]msec}
	}
};#>>>


proc Table::getSelection {} {
	variable tablew
	return [$tablew curselection]
}

proc Table::getActive {} {
	variable tablew
	if {[catch {$tablew index active} index]} {
		return ""
	}
	set r [$tablew index $index row]
	set c [$tablew index $index col]
	dputs {active=$index, r=$r, c=$c}
	if {[$tablew cget -rows] < $r} {return ""}
	if {[$tablew cget -cols] < $c} {return ""}
	return $index
}

proc Table::getRow {} {
	variable tablew
	return [$tablew cget -rows]
}

proc Table::getColumn {} {
	variable tablew
	return [$tablew cget -cols]
}

proc Table::see {{index ""}} {
	variable tablew
	if {$index eq ""} {
		set active [$tablew tag cell active]
		if {[scan $active {%d,%d} row col] != 2} {return}
		if {$row < 1 || $col < 1} {return}
		set index $row,$col
	}
	set sel [lindex $index 0]
	$tablew selection clear all
	$tablew selection set $sel $sel
	$tablew activate $sel
	$tablew see $sel
}

# delete selected row
proc Table::deleteRow {{deltag sel}} {
	variable tablew
	variable editable
	variable edit
	variable rowid
	variable rawdata
	dputs {Table::deleteRow}
	
	if {!$editable} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "You can't modify the table in view mode."]
		return
	}
	::update
	set query "delete from $edit(db.table) where rowid in "
	set dlist [list]
	set rlist [$tablew tag row $deltag]
	foreach r $rlist {
		lappend dlist "$rowid($r,1)"
	}
	append query "([join $dlist ,]);"
	set oldcount $::Sqlite::total_changes
	if {[Sqlite::evalQuery $query] == 1} { return }
	set count [expr {$::Sqlite::total_changes - $oldcount}]

	set cmax [$tablew cget -cols]
	set colrange [Util::range 1 [expr {$cmax -1}]]
	$tablew configure -state normal
	foreach tdata [lsort -integer -decr -index 0 \
		[Util::exactConsecutiveNumber $rlist]] {
		foreach {index num} $tdata {
			# table
			$tablew delete rows -- $index $num

			# rowid and rawdata
			set max [$tablew cget -rows]
			if {[Sqlite::getCurrentVersion] > 2 && 
				$::database(encoding) ne "utf-8"} {
				# rowid and rawdata
				for {set r1 $index; set r2 [expr {$index+$num}]} \
					{$r2 <= $max} {incr r1; incr r2} {
					set rowid($r1,1) $rowid($r2,1)
					foreach col $colrange {
						set rawdata($r1,$col) $rawdata($r2,$col)
					}
				}
			} else {
				# rowid only
				for {set r1 $index; set r2 [expr {$index+$num}]} \
					{$r2 <= $max} {incr r1; incr r2} {
					set rowid($r1,1) $rowid($r2,1)
				}
			}
		}
	}
	dputs {remove from table done}
	$tablew configure -state disabled

	Navi $tablew center
;#    if {[Sqlite::getCurrentVersion] == 3 && $count > [llength $rlist]} {
;#        ::Message::show -type apperror -textbox 0 \
;#        -message [msgcat::mc "More than one rows were updated by trigger.  Data of table may not be synchronized.  Please re-open this table for synchronizing."]
;#    }
	::update
	updateDataEditor
}

# insert new row
proc Table::insertRow {} {
	variable edit
	variable editable
	if {!$editable} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "You can't modify the table in view mode."]
	} else {
		RowEditor::begin $edit(db) $edit(table) $edit(colnames) $edit(col.autoincr)
	}
	::update
}

proc Table::updateRow {index value {blob 0}} {
	variable tablew
	variable table
	variable rowid
	variable rawdata
	variable edit
	variable editable

	if {!$editable} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "You can't modify the table in view mode."]
		return
	}
	if {[scan $index %d,%d row col] != 2} return
	if {$table($index) eq $value} return

	set colname [lindex $edit(colnames) [expr {$col-1}]]
	set rowidn $rowid($row,1)
	if {$blob} {
		if {[Sqlite::getCurrentVersion] == 2} {
			# value must be encoded by base64
			set query "UPDATE $edit(db.table) SET \"$colname\" = '$value' WHERE rowid = $rowidn;"
			set options [list]
		} else {
			binary scan $value H* hexvalue
			set hexvalue X'$hexvalue'
			set query "UPDATE $edit(db.table) SET \"$colname\" = $hexvalue WHERE rowid = $rowidn;"
			set options [list]
		}
	} else {
		set qvalue [string map {' ''} $value]
		set query "UPDATE $edit(db.table) SET \"$colname\" = '$qvalue' WHERE rowid = $rowidn;"
		set options [list]
	}
	set oldcount $::Sqlite::total_changes
	if {[::Sqlite::evalQuery $query $options] == 0} {
		set count [expr {$::Sqlite::total_changes - $oldcount}]

		set table($index) $value

		if {[Sqlite::getCurrentVersion] > 2 && 
			$::database(encoding) ne "utf-8"} {
			set rawvalue $value
			if {!$blob} {
				Cmd::encodingConvert to rawvalue
			}
			set rawdata($index) $rawvalue
		}
		
		# if this row was integer primary key, the column is
		# same as rowid. We must sync Table::rowid data.
		set query "SELECT rowid FROM $edit(db.table) WHERE rowid = $rowidn;"
		if {[::Sqlite::evalQuery $query] == 1} {#error}
		if {$::Sqlite::data eq {}} {
			set rowid($row,1) $value
		}
		updateDataEditor
	}
}

# Set table property editable or not
proc Table::setEditable {db table editable} {
	variable tablew
	variable edit
	dputs {Table::setEditable db=$db, table=$table, editable=$editable}

	if {$editable} {
		set ::Table::editable 1
		set edit(db) $db
		set edit(table) $table
		set edit(db.table) "'$db'.'$table'"
		set query "SELECT * FROM '$db'.'$table' WHERE rowid IS NULL"
		if {[::Sqlite::getColumnNamesFromEval $query] == 1} {
			setEditable {} {} 0
			return 1
		}
		set edit(colnames) $::Sqlite::data
		# index of integer primary key.
		# This row is auto incremented.
		set edit(col.autoincr) [Sqlite::getAutoIncrColumn $db $table index]

		bind $tablew <Double-1> [namespace code {
			if {[$tablew index @%x,%y] eq \
				[$tablew index active]} {
				::Table::CellEditor::begin \
					[$tablew index active] \
					$::Table::edit(db.table) \
					$::Table::edit(colnames)
			}
		}]
		bind $tablew <Key-Return> [namespace code {
			Navi $tablew center
			CellEditor::begin \
				[$tablew index active] \
				$::Table::edit(db.table) \
				$::Table::edit(colnames)
		}]
		.table.pop entryconfigure [msgcat::mc "Insert New Row"] -state normal
		.table.pop entryconfigure [msgcat::mc "Delete Row"]     -state normal
	} else {
		set ::Table::editable 0
		set edit(db) ""
		set edit(table) ""
		set edit(db.table) ""
		$tablew configure -state disabled 
		bind $tablew <Double-1> {}
		bind $tablew <Key-Return> {}
		.table.pop entryconfigure [msgcat::mc "Insert New Row"] -state disabled
		.table.pop entryconfigure [msgcat::mc "Delete Row"]     -state disabled
	}
}


proc Table::edit {db tablename} {
	variable tablew
	variable table
	variable rowid
	variable rawdata

	setEditable $db $tablename 1
	set tname "'$db'.'$tablename'"

	if {[::Sqlite::publish "SELECT rowid FROM $tname" rowid {}] == 1} {
		return 1
	}
	$tablew configure -variable ""
	if {[::Sqlite::publish "SELECT * FROM $tname" table rawdata] == 1} {
		$tablew configure -variable ::Table::table
		return 1
	}
	$tablew configure -variable ::Table::table
	updateDataEditor
	return 0
}

proc Table::show {sql} {
	variable tablew
	variable table
	variable rawdata

	setEditable {} {} 0
	$tablew configure -variable ""
	if {[::Sqlite::publish $sql table rawdata] == 1} {
		$tablew configure -variable ::Table::table
		return 1
	}
	$tablew configure -variable ::Table::table
	updateDataEditor
	return 0
}


proc Table::setTitle {titleStr icon} {
	variable title
	$title.label configure -text $titleStr -image $icon
}


proc Table::clear {} {
	variable tablew
	variable title
	variable table
	variable rowid
	variable rawdata

	if {$CellEditor::info(start)} {CellEditor::end 1}
	if {$RowEditor::info(start)} {RowEditor::end 1}

	set focusW [focus]
	set cedit $::Table::CellEditor::root.text
	set redit $::Table::RowEditor::root.text
	if {$focusW eq $cedit || $focusW eq $redit} {
		focus -force $tablew
	}
	array unset table;  array set table {}
	array unset rowid;  array set rowid {}
	array unset rawdata; array set rawdata {}
	
	$tablew configure -rows 1 -cols 1
	$title.label configure -text [msgcat::mc Table] -image {}
	updateDataEditor
	::update
}


proc Table::update {} {
	variable tablew
	variable editable
	variable edit
	dputs {Table::update}
	if {$editable} {
		set index [$tablew index active]
		edit $edit(db) $edit(table)
		$tablew activate $index
	}
	updateDataEditor
}

# set list value
proc Table::getList {rowsName colsName listName} {
	variable tablew
	variable table
	upvar $listName data
	upvar $rowsName rows
	upvar $colsName cols
	set rmax [$tablew cget -rows]
	set cmax [$tablew cget -cols]
	set rows $rmax
	set cols [expr {$cmax-1}]
	if {$cols < 1} {return}
	set crange [Util::range 1 [incr cmax -1]]

	for {set row 0} {$row < $rmax} {incr row} {
		set rowdata [list]
		foreach col $crange {
			lappend data $table($row,$col)
		}
	}
}

proc Table::updateDataEditor {} {
	variable tablew
	variable table
	variable rowid
	variable editable
	variable edit
	variable rawdata
	
	set active [$tablew tag cell active]
	if {[Tool::DataEditor::getStatus] ne {show}} {return}
	if {[scan $active {%d,%d} row col] != 2} {return}
	if {$row < 1 || $col < 1} {
		# Table is editable but has no data.
		Tool::DataEditor::setInfo \
			{} \
			{} \
			{} \
			{} \
			0
		Tool::DataEditor::setRawData {}
		return
	}

	if {$editable} {
		if {[array names rowid $row,1] ne ""} {
			set rowidn $rowid($row,1)
		} else {
			set rowidn ""
		}
		Tool::DataEditor::setInfo \
			$edit(db.table) \
			$rowidn \
			$active \
			$table(0,$col) \
			$editable
	} else {
		Tool::DataEditor::setInfo \
			$edit(db.table) \
			"" \
			$active \
			$table(0,$col) \
			$editable
	}
	if {[array names table $row,$col] eq ""} {
		set data {}
	} else {
		if {[Sqlite::getCurrentVersion] > 2 &&
			$::database(encoding) ne "utf-8"} {
			set data $rawdata($row,$col)
		} else {
			set data $table($row,$col)
		}
	}
	Tool::DataEditor::setRawData $data
}
;#>>>

;#>>>


namespace eval History {;#<<<
	variable info
	array set info {
		max 100
		val {}
	}
}

# setup widgets
proc History::History {} {
	ttk::frame .history -height 100
	
	set title [Titlebar::Titlebar .history.title [msgcat::mc "Command History"] {}]
	pack $title -side top -fill x
	
	ttk::frame .history.f -style TEntry -border 2
	grid $title -row 0 -column 0 -sticky ew
	grid .history.f -row 1 -column 0 -sticky news
	grid columnconfigure .history 0 -weight 1
	grid rowconfigure    .history 1 -weight 1
	
	ttk::scrollbar .history.f.y -command {.history.f.list yview}
	listbox .history.f.list -relief flat -takefocus 1 \
		-highlightthickness 0 \
		-listvariable ::History::info(val) \
		-yscrollcommand {.history.f.y set}
	grid .history.f.y    -row 0 -column 1 -sticky ns
	grid .history.f.list -row 0 -column 0 -sticky news
	grid columnconfigure .history.f 0 -weight 1
	grid rowconfigure    .history.f 0 -weight 1
	
	menu .history.pop
	.history.pop add command -label [msgcat::mc Delete]    -command {::History::deleteSelection}
	.history.pop add command -label [msgcat::mc "Delete All"] -command {::History::deleteAll}

	bind .history.f.list <1> {focus -force %W}
	bind .history.f.list <Double-1> {Cmd::doubleClickHistory}
	bind .history.f.list <<B3>> {focus -force %W; tk_popup .history.pop %X %Y; break}
	Cmd::bindTextFocusTtkEntry .history.f.list .history.f
	
	return .history
}

proc History::append {command} {
	variable info
	if {$command ne [lindex $info(val) end]} {
		lappend info(val) $command
		set len [llength $info(val)]
		if {$len > $info(max)} {
			set info(val) [lrange $info(val) [expr {$len - $info(max)}] end]
		}
	}
	.history.f.list selection clear 0 end
	.history.f.list selection anchor end
	.history.f.list selection set end
	.history.f.list see end
}

proc History::deleteSelection {} {
	variable info
	set sel [.history.f.list curselection]
	if {$sel eq {}} return
	set info(val) [lreplace $info(val) $sel $sel]
}

proc History::deleteAll {} {
	variable info
	set info(val) {}
}

proc History::getSelection {} {
	variable info
	return [lindex $info(val) [.history.f.list curselection]]
}

proc History::dump {} {
	variable info
	set session "#History::dump\n"
	::append session "array set [namespace current]::info \[list [array get info]\]\n"
	::append session ".history.f.list see end\n"
	return $session
}
;#>>>


namespace eval Sqlite {;#<<<
	variable data
	variable total_changes 0 ;# only sqlite3.
	array set status {
		haveSqlite2 0
		haveSqlite3 0
		time 0
		lines 0
	}
	array set info {
		v3,legacyfileformat 1 
		v3,pagesize         1024
		v3,encoding         "UTF-8"
		extension           {}
	}
	array set interp {2 {} 3 {} current {}}

	array set master {}
	# [list ? {seq name file} ...?]
	# A first element must be main.
	# if llength == 0 then database was closed.
	set master(database) [list]
	;# this is [list ? {databasename type name tbl_name rootpage sql} ... ?]
	set master(table)	[list]
	;# this is [list ? {databasename type name tbl_name rootpage sql} ... ?]
	set master(view)	[list]
		# NOTE:
		# tclsqlite2 is ISO8859 and tclsqlite3 is UTF-8.
}

# return current sqlite interp
proc Sqlite::getInterp {} {
	variable interp
	return $interp(current)
}

# return current sqlite version 2 or 3 or {}
proc Sqlite::getCurrentVersion {} {
	variable interp
	if {$interp(current) eq ""} {return {}}
	return [$interp(current) eval set version]
}

# return  current sqlite full version number 
proc Sqlite::getCurrentFullVersion {} {
	variable interp
	if {$interp(current) eq ""} {return {}}
	return [db eval {SELECT sqlite_version();}]
}

# create new
proc Sqlite::create {version} {
	set filename [tk_getSaveFile -filetypes $::pref(openTypeSqlite)]
	if { $filename eq {} } {return}
	file delete -force -- $filename
	open $filename $version
	return $filename
}

# search file exists and return file version 
# if we can't open it, throw error
proc Sqlite::tryOpen {filename version} {
	variable status
	if {![file exists $filename]} {
		return -code error \
		$filename\n[msgcat::mc "The file doesn't exist."]
	}
	
	if {[file size $filename] == 0} {
		if {$version eq {}} {
			return -code error \
			$filename\n[msgcat::mc "Faied to open the file. The file has zero size."]
		}
	} else {
		set version [Sqlite::getFileVersion $filename]
	}

	switch -exact -- $version {
		2 {
			if {$status(haveSqlite2) == 0} {
				return -code error \
				[msgcat::mc "Please install tclsqlite, if you want to open SQLite2 file."]
			}
		}
		3 {
			if {$status(haveSqlite3) == 0} {
				return -code error \
				[msgcat::mc "Please install tclsqlite3, if you want to open SQLite3 file."]
			}
		}
		default {
			return -code error \
			$filename\n[msgcat::mc "The file is not a SQLite file."]
		}
	}
	return -code ok $version
}

# open
proc Sqlite::open {filename version} {
	variable total_changes
	variable master
	variable data
	variable info
	variable cmd
	variable interp
	
	switch -exact -- $version {
		2 { interp alias {} sqlite $interp(2) sqlite }
		3 { interp alias {} sqlite $interp(3) sqlite3 }
		default {return -code error "version is $version"}
	}
	set interp(current) $interp($version)
	
	# set database statusmation
	set ::database(path) $filename
	set ::database(name) [file tail $filename]
	set ::database(version) $version

	cd [file dirname $filename]

	sqlite db [file tail $filename]
	interp alias {} db $interp($version) db
	set data {}
	if {[getCurrentVersion] == 3} {
		updateInfo
		set total_changes [db total_changes]
		catch {db enable_load_extension 1}
	}
	installFunctions $interp($version)
	loadExtension $interp($version)
	return 0
}


proc Sqlite::export {db fp} {
	variable data
	variable interp
	if {$interp(current) eq {}} {return {}}
	set master [getSqliteMasterName $db]

	puts $fp "BEGIN TRANSACTION;"

	# Table
	evalQuery "select name, sql from $master where type = 'table'"
	foreach {name sql} $data {
		puts $fp "$sql;"

		set query "select 'INSERT INTO ' ||  '\"$name\"' ||  ' VALUES('  || "
		getColumnNamesFromEval "select * from '$db'.'$name' where rowid = NULL"
		set collist [list]
		foreach col $data {
			lappend collist "quote(\"$col\")"
		}
		append query [join $collist " || \", \" || "]
		append query " || ');' from '$db'.'$name';"

		evalQuery $query
		foreach stat $data { puts $fp $stat }
	}
	
	# Index
	evalQuery "select sql from $master where type = 'index' and sql not null"
	foreach sql $data { puts $fp "$sql;" }
	
	# View
	evalQuery "select sql from $master where type = 'view' and sql not null"
	foreach sql $data { puts $fp "$sql;" }

	# Trigger
	evalQuery "select sql from $master where type = 'trigger' and sql not null"
	foreach sql $data { puts $fp "$sql;" }

	# End
	puts $fp "COMMIT;"
}


proc Sqlite::import {query} {
	variable interp
	if {$interp(current) eq {}} {return {}}

	Cmd::encodingConvert to query
	prepareSysEncodingForSqlite2

	if {[catch {db eval $query} msg]} {
		restoreSysEncodingForSqlite2
		#check transaction
		if {![catch {db eval "begin;"}]} {
			db eval "rollback;"
			::Message::show -type apperror \
			 -message [msgcat::mc "Failed to import SQL."] -detail $msg
			return 1
		}

		set result [::Message::show -type apperror -buttons {rollback commit nothing} \
		-default rollback -message [msgcat::mc "Failed to import SQL."] \
		-labels {rollback Rollback commit Commit nothing "Do Nothing"} -detail $msg]
		
		switch -exact -- $result {
			rollback {evalQuery "rollback;"}
			commit   {evalQuery "commit;"}
			nothing  {}
		}
		return 1
	}
	restoreSysEncodingForSqlite2
	return 0
}

proc Sqlite::getSqliteMasterName {db} {
	if {$db eq "temp"} {
		return sqlite_temp_master
	}
	return "'$db'.sqlite_master"
}

# return auto increment colname. it is a integer primary key.
# rettype : name  or index
proc Sqlite::getAutoIncrColumn {db table {rettype name}} {
	variable interp
	if {$interp(current) eq {}} {return {}}
	
	set i 0
	set rlist [list]
	foreach cdata [getColumnList $db $table] {
		dict with cdata {};# set values
		
		if {$pkey == 1} {
			if {[string compare -nocase $type "integer"] == 0} {
				lappend rlist [list $name $i]
			}
		}
		incr i
	}
	# If there were some integer primary keys, the cols is not
	# auto incremented. so return empty.
	if {[llength $rlist] == 1} {
		switch -exact -- $rettype {
			name  {return [lindex $rlist 0 0]}
			index {return [lindex $rlist 0 1]}
		}
	}
	return
}


# Return {dbname type ?dbname type ...?}
# Note: Sqlite can't get same named table infomation correctly.
# So we must check is there same named table, before access the table.
proc Sqlite::findTable {tablename {versionCheck 0}} {
	variable data
	variable interp
	set version $interp(current)

	set ret [list]
	foreach db [getDBList all] {
		set master [getSqliteMasterName $db]
		set query "select name, type from $master where (type = 'table' or type = 'view') and name = '$tablename' order by name;"
		if {[evalQuery $query] != 0} { return {} }
		foreach {name type} $data {
			lappend ret $db $type
		}
	}
	if {$versionCheck == 1 && $version == 3} {
		set findTemp 0
		foreach {db type} $ret {
			if {$db eq "temp"} {
				set findTemp 1
				break
			}
		}
		if {$findTemp == 0} {return {}} 
	}
	return $ret
}


# return dblist names. 
# type : all, writable
proc Sqlite::getDBList {{type all}} {
	variable data
	set dblist [list]
	
	if {$type eq "writable"} {
		if {[getCurrentVersion] == 3} {
			if {[evalQuery {PRAGMA database_list;}] == 0} {
				foreach {seq name file} $data {
				lappend dblist $name
				}
			}
			if {[lsearch -exact $dblist "temp"] < 0} {
				set dblist [linsert $dblist 1 "temp"]
			}
		} else {
			set dblist [list main temp]
		}
	} else {
		if {[evalQuery {PRAGMA database_list;}] == 0} {
			foreach {seq name file} $data {
			lappend dblist $name
			}
		}
		if {[lsearch -exact $dblist "temp"] < 0} {
			set dblist [linsert $dblist 1 "temp"]
		}
	}
	return $dblist
}


# return tablelist
proc Sqlite::getTableList {db} {
	variable data
	set master [getSqliteMasterName $db]
	set query "select name from $master where type = 'table' order by name"
	if {[evalQuery $query] == 0} {
		return $data
	}
	return [list]
}

# return viewlist
proc Sqlite::getViewList {db} {
	variable data
	set master [getSqliteMasterName $db]
	set query "select name from $master where type = 'view' order by name"
	if {[evalQuery $query] == 0} {
		return $data
	}
	return [list]
}

# return column list of dict name type null def pkey defisnull unique
# if version > 3.3.7
# def value of table_info is quoted with "'", if it is string.
proc Sqlite::getColumnList {db table} {
	variable interp
	if {$interp(current) eq {}} {return {}}
	
	# Find unique column
	set uniquecolumn [dict create]
	set ilist [getIndexList $db $table]
	foreach index $ilist {
		foreach {name unique colnames} $index {break}
		if {[string match {sqlite_*} $name]} {
			foreach col $colnames {
				dict set uniquecolumn $col 1
			}
		}
	}

	if {[getCurrentVersion] == 2} {
		# Version 2
		# In sqlite2 we can't use [db nullvalue]. 
		# we must parse the sql for distinguish between NULL and Empty String.
		set query "PRAGMA table_info('$table');"
		Cmd::encodingConvert to query
		prepareSysEncodingForSqlite2
		if {[catch {db eval $query} result]} {
			restoreSysEncodingForSqlite2
			Cmd::encodingConvert from query
			error $query\n$result
		}
		set clist [list]
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from result 1

		# parse the sql
		set master [getSqliteMasterName $db]
		set query "SELECT sql FROM $master WHERE type = 'table' AND tbl_name = '$table';"
		Cmd::encodingConvert to query
		prepareSysEncodingForSqlite2
		set sql [lindex [db eval $query] 0]
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from sql

		if {$sql eq ""} {
			# View 
			set query "PRAGMA table_info('$table');"
			Cmd::encodingConvert to query
			prepareSysEncodingForSqlite2
			set result [db eval $query]
			restoreSysEncodingForSqlite2
			Cmd::encodingConvert from result
			# TODO:we have no way to know details of columns of view.
			set clist [list]
			foreach {cid name type null def pkey} $result {
				set unique [dict exists $uniquecolumn $name]
				lappend clist [dict create name $name type $type null $null \
					def $def defisnull 0 pkey $pkey unique $unique]
			}
			return $clist
		}

		set tabledata [::SQLParser::parseCreateTable [::SQLParser::removeComment $sql]]
		set i 0
		foreach {cid name type null def pkey} $result {
			set unique [dict exists $uniquecolumn $name]
			# check def is null
			set defisnull 1
			if {[dict exists $tabledata column]} {
			set coldata [dict get $tabledata column $i]
				if {[dict exists $coldata constraint]} {
					set const [dict get $coldata constraint]
					dict for {n constdata} $const {
						if {[dict get $constdata type] eq "default"} {
							set defisnull 0
							break
						}
					}
				}
			}
			lappend clist [dict create name $name type $type null $null \
				def $def defisnull $defisnull pkey $pkey unique $unique]
			incr i
		}
	} else {
		# Version 3~
		# In Tcl, we can't distinguish between NULL and EmpyString. 
		# so change nullvalue twice, and find the NULL.
		set query "PRAGMA '$db'.table_info('$table');"
		Cmd::encodingConvert to query
		set tempnullvalue [db nullvalue]
		foreach nn {1 2} {
			db nullvalue $nn
			if {[catch {db eval $query} result$nn]} {
				Cmd::encodingConvert from query
				db nullvalue $tempnullvalue
				dputs {$query\n[set result$nn]}
				return [list]
			}
		}

		foreach {_v2 _v2temp _v3 version} [getLibraryVersion] {break}

		set clist [list]
		Cmd::encodingConvert from result1 1
		foreach {cid1 name1 type1 null1 def1 pkey1} $result1 \
				{cid2 name2 type2 null2 def2 pkey2} $result2 {
			set defisnull [expr {$def1 ne $def2}]
			if {$defisnull} {set def1 {}}
			set unique [dict exists $uniquecolumn $name1]

			lappend clist [dict create name $name1 type $type1 null $null1 \
				def $def1 defisnull $defisnull pkey $pkey1 unique $unique]
		}
		db nullvalue $tempnullvalue
	}

	return $clist
}


# return index list
proc Sqlite::getIndexList {db table} {
	variable interp
	if {[getCurrentVersion] == 3} {
		return [getIndexListFromPragma $db $table]
	}
	return [getIndexListForSqlite2 $db $table]
}
proc Sqlite::getIndexListForSqlite2 {db table} {;#<<<
	variable data
	variable interp
	if {[getCurrentVersion] ne 2} {return {}}

	# check for another attached database.
	set dblist [getDBList]
	if {[llength $dblist] == 2} {
		# There is only main and temp.
		return [getIndexListFromPragma $db $table]
	}
	# get same named indexlist of the table have
	set master [getSqliteMasterName $db]
	set query "select name from $master where type = 'index' and tbl_name = '$table';"
	if {[evalQuery $query] != 0} { return [getIndexListFromPragma $db $table] }
	set inamelist $data
	if {[llength $inamelist] == 0} {return {}}
	set wnames [list]
	foreach in $inamelist {
		lappend wnames "name = '$in'"
	}
	set querylist [list]
	foreach adb $dblist {
		set master [getSqliteMasterName $adb]
		lappend querylist "select name, sql, '$adb' as db from $master\nwhere type = 'index' and ([join $wnames { or }])"
	}
	set query [join $querylist "\nunion\n"]
	if {[evalQuery $query] != 0 || [llength $data] == 3} {
		return [getIndexListFromPragma $db $table]
	}
	# try to find same name index
	set sameindex $data
	set ret [list]
	foreach iname $inamelist {
		set existSame 0
		set sql {}
		foreach {sindex ssql sdb} $sameindex {
			if {$sindex eq $iname} {
				if {$sdb eq $db} {
					set sql $ssql;
				} else {
					set existSame 1;
				}
			}
		}
		if {$existSame} {
			if {$sql eq {}} {
				lappend ret [list $iname 1 {}];#sqlite_autoindex
			} else {
				set indexdict [SQLParser::parseCreateIndex2 [SQLParser::removeComment $sql]]
				if {[dict size $indexdict] == 0} {
					lappend ret [list $iname 1 {}]
				} else {
					set _name [dict get $indexdict index]
					set _uniq [dict get $indexdict unique]
					set _cols ""
					foreach {n coldict} [dict get $indexdict columns] {
						lappend _cols [dict get $coldict name]
					}
					lappend ret [list $iname $_uniq $_cols]
				}
			}
		} else {
			set query "PRAGMA index_info('$iname')"
			Cmd::encodingConvert to query
			prepareSysEncodingForSqlite2
			if {[catch {db eval $query} coldata]} {
				restoreSysEncodingForSqlite2
				Cmd::encodingConvert from query
				error $query\n$coldata
			}
			set colnames [list]
			restoreSysEncodingForSqlite2
			Cmd::encodingConvert from coldata 1
			foreach {seqno cid colname} $coldata {
				lappend colnames $colname
			}
			if {$sql eq {}} {
			set uniq 1 ;#sqlite_autoindex
			} else {
			set uniq [regexp -nocase -- {^\s*CREATE\s+INDEX\s+UNIQUE} $sql]
			}
			lappend ret [list $iname $uniq $colname]
		}
	}
	return $ret
};#>>>
proc Sqlite::getIndexListFromPragma {db table} {;#<<<
	variable interp
	set version [getCurrentVersion]
	if {$interp(current) eq {}} {return {}}
	switch -exact -- $version {
		2 { set query "PRAGMA index_list('$table')" }
		3 { set query "PRAGMA '$db'.index_list('$table')" }
	}

	Cmd::encodingConvert to query
	prepareSysEncodingForSqlite2
	if {[catch {db eval $query} result]} {
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from query
		error $query\n$result
	}

	set ilist [list]
	restoreSysEncodingForSqlite2
	Cmd::encodingConvert from result 1
	foreach {num name unique} $result {
		switch -exact -- $version {
			2 { set query "PRAGMA index_info('$name')" }
			3 { set query "PRAGMA '$db'.index_info('$name')" }
		}

		Cmd::encodingConvert to query
		prepareSysEncodingForSqlite2
		if {[catch {db eval $query} data]} {
			restoreSysEncodingForSqlite2
			Cmd::encodingConvert from query
			error $query\n$data
		}
		set colnames [list]
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from data 1
		foreach {seqno cid colname} $data {
			lappend colnames $colname
		}
		lappend ilist [list $name $unique $colnames]
	}
	return [lsort -dictionary -index 0 $ilist]
};#>>>
	

# return trigger list {name sql}
proc Sqlite::getTriggerList {db table} {
	variable interp
	if {$interp(current) eq {}} {return {}}
	# type name tbl_name rootpage sql
	set master [getSqliteMasterName $db]
	set query "select name, sql from $master where tbl_name='$table' and type='trigger'"

	Cmd::encodingConvert to query
	prepareSysEncodingForSqlite2
	if {[catch {db eval $query} result]} {
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from query
		error $query\n$result
	}

	set tlist [list]
	restoreSysEncodingForSqlite2
	Cmd::encodingConvert from result 1
	foreach {name sql} $result {
		lappend tlist [list $name $sql]
	}
	return [lsort -dictionary -index 0 $tlist]
}

# return collation list. 1st element of the list
# must be empty list.
proc Sqlite::getCollationList {db} {
	variable interp
	if {$interp(current) eq {}} {return {}}

	switch -exact -- [getCurrentVersion] {
		3 {
			set query "PRAGMA collation_list"
			Cmd::encodingConvert to query
			if {[catch {db eval $query} result]} {
				Cmd::encodingConvert from query
				error $query\n$result
			}
			set collations [concat [list {}] [dict values $result]]
		}
		2 -
		default {
			set collations [list {} BINARY NOCASE]
		}
	}
	return $collations
}

# column title from sql result, return 0 | 1
# set result to Sqlite::data
proc Sqlite::getColumnNamesFromEval {query} {
	variable data
	variable interp
	if {$interp(current) eq {}} {
		return 1
	}
	set cinterp $interp(current)
	$cinterp eval {set colTitle {}}

	Cmd::encodingConvert to query
	prepareSysEncodingForSqlite2
	db eval "PRAGMA empty_result_callbacks = on"
	$cinterp eval {set val(*) {}}
	if {[catch {db eval $query val {}} msg]} {
		set data [list]
		restoreSysEncodingForSqlite2
		return 1
	}

	set colnames [$cinterp eval {return $val(*)}]
	restoreSysEncodingForSqlite2
	Cmd::encodingConvert from colnames 1
	set data $colnames
	return 0
}


# Do eval sql and set result to Sqlite::data, if got error return 1
proc Sqlite::evalQuery {query {optionList {}}} {
	variable data
	variable total_changes
	variable info
	variable interp
	array set option {
		encode   1
		time     0
		varNames {}
	}
	array set option $optionList

	if {$interp(current) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return 1
	}
	dputs {$query}
	
	db eval "PRAGMA count_changes=on"
	
	if {$option(encode)} { Cmd::encodingConvert to query }
	set st 0
	set ed 0
	if {[llength $option(varNames)] > 0} {
		foreach {varName} $option(varNames) {
			if {[info var $varName] ne ""} {
				error "can't use this varName: $varName"
			}
			upvar $varName $varName
		}
	}

	prepareSysEncodingForSqlite2
	set st [clock clicks -milliseconds]
	
	catch {db eval $query} data
	if {[Sqlite::hasError]} {
		restoreSysEncodingForSqlite2
		if {$option(encode)} {
			Cmd::encodingConvert from query
			Cmd::encodingConvert from data
		}
		::Message::show -type apperror \
		-message [msgcat::mc "Invalid sql query"] -detail $data\n\n$query
		set status(time) 0
		return 1
	}
	set ed [clock clicks -milliseconds]
	restoreSysEncodingForSqlite2
	
	
	if {[getCurrentVersion] != 2} {
		set total_changes [db total_changes]
	}

	if {$option(time)} {
		set status(time) [expr {$ed - $st}]
	}
	if {$option(encode)} {
		Cmd::encodingConvert from data 1
	}
	return 0
}


# Do eval sql and Update table array 
# This proc is colled from Table only.
# time measure.
# if SQLite3 and not utf-8 encoding, set rawdata to rawdataName,
proc Sqlite::publish {query tableName rawdataName} {
	upvar $tableName table 
	upvar $rawdataName rawdata
	variable data
	variable total_changes
	variable status
	variable interp
	if {$interp(current) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return 1
	}
	dputs {$query}

	set cinterp $interp(current)
	# Get DataList
	Cmd::encodingConvert to query
	prepareSysEncodingForSqlite2
	set st [clock clicks -milliseconds]
	if {[catch {db eval $query} data]} {
		restoreSysEncodingForSqlite2
		Cmd::encodingConvert from query
		Cmd::encodingConvert from data
		::Message::show -type apperror \
		-message [msgcat::mc "Invalid sql query"] -detail $data\n\n$query
		set status(time) 0
		return 1
	}
	set ed [clock clicks -milliseconds]
	set status(time) [expr {$ed - $st}]
	restoreSysEncodingForSqlite2

	if {[getCurrentVersion] == 3} {
		set total_changes [db total_changes]
	}
	
	set _temp [string map {, { } \" {}} $query]
	set _temp [SQLParser::removeComment $_temp]
	set sqlcmd [string tolower [lindex $_temp 0]]
	if {$sqlcmd ne "select" && 
		$sqlcmd ne "pragma" &&
		$sqlcmd ne "explain"} {
		.table.f.table configure -rows 1 -cols 1
		return 0
	}

	# Get Title
	db eval "PRAGMA empty_result_callbacks = on"
	$cinterp eval {set val(*) {}}

	prepareSysEncodingForSqlite2
	db eval $query val {break}
	restoreSysEncodingForSqlite2

	$cinterp eval {set colTitle {}}
	set titlelist [$cinterp eval {return $val(*)}]
	Cmd::encodingConvert from titlelist 1

	# Copy values from slave to master
	array unset table
	array set table {}

	set i 1
	foreach colname $titlelist {
		set table(0,$i) $colname
		incr i
	}
	
	set collen [llength $titlelist]
	set col 1
	set row 1
	if {[string length $rawdataName] > 0 && 
			[getCurrentVersion] > 2 && 
			$::database(encoding) ne "utf-8"} {
		# here is SQLite3 and NOT utf-8 only
		array unset rawdata
		array set rawdata {}
		set rawdatalist $data
		Cmd::encodingConvert from data 1
		foreach d $data rd $rawdatalist {
			set table($row,$col) $d
			set rawdata($row,$col) $rd
			if {($col % $collen) == 0} {
				set col 0
				incr row
			}
			incr col
		}
		unset rawdatalist
	} else {
		Cmd::encodingConvert from data 1
		foreach d $data {
			set table($row,$col) $d
			if {($col % $collen) == 0} {
				set col 0
				incr row
			}
			incr col
		}
	}
	
	# TODO:move this code to ::Table namespace or 
	# call something like update.
	.table.f.table configure -rows $row \
		-cols [expr {1 + [llength $titlelist]}]

	return 0
}

# if query is not utf-8, we must change the encodings.
# if called prepareQuery, restoreQuery MUST be called after.
proc Sqlite::prepareSysEncodingForSqlite2 {} {
	variable interp
	switch -exact -- $::database(version) {
		2 {$interp(current) eval encoding system $::database(encoding)}
		3 {}
	}
	set ::Sqlite::func::encoding $::database(encoding)
}
proc Sqlite::restoreSysEncodingForSqlite2 {} {
	variable interp
	switch -exact -- $::database(version) {
		2 {$interp(current) eval encoding system $::SYSENCODING}
		3 {}
	}
}

proc Sqlite::getLastInsertRowid {} {
	variable interp
	if {$interp(current) eq {}} {
		return 1
	}
	return [db last_insert_rowid]
}

# in 3.5.6, SQLITE_OK(0), SQLITE_ROW(100), SQLITE_DONE(101)
# has the message "not an error". But Tclsqlite interface
# returns as an error when SQLITE_ROW and SQLITE_DONE.
# So it is not enough catch {...} statement error handling.
# This proc returns true, if db has an error.
proc Sqlite::hasError {} {
	set code [db errorcode]
	return [expr {$code != 0 && $code != 100 && $code != 101}]
}

# close DB
proc Sqlite::close {} {
	variable interp
	if { $::database(name) == {} } return
	db close
	set interp(current) {}
	variable data {}
}

proc Sqlite::vacuum {} {
	variable info
	variable data

	if {[getCurrentVersion] eq "3"} {
		db eval "PRAGMA legacy_file_format = $info(v3,legacyfileformat) ;"
	}
	if {[evalQuery {VACUUM;}] == 0} {
		::Message::show -type appinfo \
		-message "SQLite Result" -detail ok
	}
}

proc Sqlite::checkIntegrity {} {
	variable data
	if {[evalQuery {PRAGMA integrity_check;}] == 0} {
		::Message::show -type appinfo \
		-message "SQLite Result" -detail ok
	}
}

# set/get default cache size or cache size
# type is "cache" or "default cache"
proc Sqlite::cache {cmd type {db "main"} {size ""}} {
	variable data
	if {[getCurrentVersion] == 2} {set pref ""} else {set pref "'$db'."}
	if {$cmd eq "set"} {
		if {$type eq "cache"} {
			set q "PRAGMA ${pref}cache_size = $size;"
		} else {
			set q "PRAGMA ${pref}default_cache_size = $size;"
		}
	}
	if {$cmd eq "get"} {
		if {$type eq "cache"} {
			set q "PRAGMA ${pref}cache_size;"
		} else {
			set q "PRAGMA ${pref}default_cache_size;"
		}
	}
	evalQuery $q
	return $data
}

proc Sqlite::pagesize {cmd {db main} {size ""}} {
	variable data
	if {[getCurrentVersion] == 2} {
		return 1024
	} else {
		set pref "\[$db\]."
	}
	if {$cmd eq "set"} {
		set q "PRAGMA ${pref}page_size = $size;"
	}
	if {$cmd eq "get"} {
		set q "PRAGMA ${pref}page_size;"
	}
	evalQuery $q
	return $data
}

proc Sqlite::updateInfo {} {
	variable info
	variable interp
	if {[lsearch {512 1024 2048 4096 8192} $info(v3,pagesize)] < 0} {
		set info(v3,pagesize) 1024
	}
	if {![string is boolean $info(v3,legacyfileformat)]} {
		set info(v3,legacyfileformat) 1
	}
	if {$interp(current) ne ""} {
	db eval "PRAGMA legacy_file_format = $info(v3,legacyfileformat);"
	db eval "PRAGMA page_size = $info(v3,pagesize);"
	db eval "PRAGMA encoding = '$info(v3,encoding)';"
	}
}

proc Sqlite::getFileVersion {file} {
	set fp [::open $file r]
	fconfigure $fp -encoding utf-8
	set line [::read $fp 47]
	::close $fp

	if {$line eq {** This file contains an SQLite 2.1 database **}} {
		return 2
	}
	if {[string match {SQLite format 3*} $line]} {
		return 3
	}
	return {}
}


proc Sqlite::getLibraryVersion {} {
	variable status
	variable interp
	set ret {}
	foreach n {2 3} {
		lappend ret $n
		if {$status(haveSqlite$n)} {
			lappend ret [$interp($n) eval {sqlite -version}]
		} else {
			lappend ret {}
		}
	}
	return $ret
}

# user functions for sqlite
# version0.5

namespace eval Sqlite::func {
	variable db ""
	variable version ""
	variable encoding ""

	# install functions to db
	proc install {dbcmd {enc ""}} {
		variable db $dbcmd
		variable version
		variable encoding
		
		set db_encoding [$db eval {PRAGMA encoding;}]
		set version  [lindex [split [$db eval "select sqlite_version();"] .] 0]
		if {$enc eq ""} {
			if {$version == 3} {
				set encoding "utf-8"
			}
			if {$version == 2} {
				set encoding [encoding system]
			}
		}
		
		foreach cmd [info command [namespace current]::*] {
			set func [namespace tail $cmd]
			if {$func eq "install"} {continue}
			if {[string match "_*" $func]} {continue}
			$db function $func $cmd
		}
	}

	proc _getUnicodeByteArray {str isLE} {
		if {$str eq ""} return
		set data [encoding convertto unicode $str]

		if {$::tcl_platform(byteOrder) eq "littleEndian"} {
			if {$isLE} {
				return $data
			} else {
				binary scan $data s* str
				return [binary format S* $str]
			}
		} else {
			if {$isLE} {
				binary scan $data s* str
				return [binary format S* $str]
			} else {
				return $data
			}
		}
	}
	proc _getStringFromUnicodeByteArray {bytearray isLE} {
		if {$bytearray eq ""} return
		if {$::tcl_platform(byteOrder) eq "littleEndian"} {
			if {!$isLE} {
				binary scan $bytearray s* data
				set bytearray [binary format S* $data]
			}
		} else {
			if {$isLE} {
				binary scan $bytearray s* data
				set bytearray [binary format S* $data]
			}
		}
		return [encoding convertfrom unicode $bytearray]
	}
	proc _getEncodedByteArray {data binary} {
		variable db
		variable version
		variable encoding
		if {$version == 3} {
			if {!$binary} {
				set db_encoding [$db eval {PRAGMA encoding;}]
				if {$db_encoding eq "UTF-8" && $encoding eq "utf-8"} {
					set data [encoding convertto "utf-8" $data]
				}
				if {$db_encoding eq "UTF-16le"} {set data [_getUnicodeByteArray $data 1]}
				if {$db_encoding eq "UTF-16be"} {set data [_getUnicodeByteArray $data 0]}
			}
		}
		if {$version == 2} {
			set data [encoding convertto identity $data]
		}
		return $data
	}
	proc _getDataFromByteArray {data binary} {
		variable db
		variable version
		variable encoding
		if {$version == 3} {
			if {!$binary} {
				set db_encoding [$db eval {PRAGMA encoding;}]
				if {$db_encoding eq "UTF-8" && $encoding eq "utf-8"} {
					set data [encoding convertfrom "utf-8" $data]
				}
				if {$db_encoding eq "UTF-16le"} {set data [_getStringFromUnicodeByteArray $data 1]}
				if {$db_encoding eq "UTF-16be"} {set data [_getStringFromUnicodeByteArray $data 0]}
			}
		}
		if {$version == 2} {
			if {!$binary} {
				set data [encoding convertfrom identity $data]
			}
		}
		return $data
	}

	proc _getTclString {data} {
		variable db
		variable version
		variable encoding
		if {$encoding ne "utf-8" && $encoding ne "unicode"} {
			return [encoding convertfrom $encoding $data]
		} else {
			return $data
		}
	}

	# String Format
	proc ascii     {a} {binary scan $a c x; return $x}
	proc char      args {set r ""; foreach c $args {append r [binary format c $c]}; return $r}
	proc concat    args {join $args {}}
	proc concat_ws {s args} {join $args $s}
	proc convert {str from to} {
		variable version
		variable db
		if {$version == 3} {
			if {$from eq "utf-8"} {
			set str [encoding convertto identity $str]
			}
			set str [encoding convertfrom $from $str]
			set str [encoding convertto $to $str]
			if {$to eq "utf-8"} {
			set str [encoding convertfrom identity $str]
			}
			return $str
		}
		if {$version == 2} {
			set str [encoding convertto identity $str]
			set str [encoding convertfrom $from $str]
			set str [encoding convertto $to $str]
			set str [encoding convertfrom identity $str]
			return $str
		}
	}
	proc elt       {n args} {lindex $args [incr n -1]}
	proc hex       {n} {format %X $n}
	proc initcap   {str} {
		set ret ""
		set len [string length $str]
		set u 1
		for {set i 0} {$i < $len} {incr i} {
			set c [string index $str $i]
			if {$u == 1} {
				set c [string toupper $c]
				set u 0
			} else {
				set c [string tolower $c]
			}
			if {[string is space $c]} {
				set u 1
			}
			append ret $c
		}
		return $ret
	}
	proc insert    {s pos len ns} {string replace $s [expr {$pos-1}] [expr {$pos-2+$len}] $ns} ;#MySQL insert
	proc instr     {str sstr {st 0} {n 1}} {
		if {$n < 1} {return 0}
		if {$st >= 0} {
			incr st -1
			for {set i 0} {$i < $n} {incr i} {
				set st [string first $sstr $str $st]
				if {$st == -1} {return 0}
				incr st
			}
		} else {
			set st [expr {[string length $str] + $st - 1}]
			for {set i 0} {$i < $n} {incr i} {
				set st [string last $sstr $str $st]
				if {$st == -1} {return 0}
			}
			incr st
		}
		return $st
	}
	proc locate    {sstr str pos} {instr $str $sstr $pos 1}
	proc position  {sstr in str} {instr $str $sstr}
	proc left      {s n} {string range $s 0 [incr n -1]}
	#    length    Sqlite built-in
	#    lower     Sqlite built-in
	proc lpad      {s n {p " "}} {
		if {$p eq {}} {set p " "}
		set slen 0
		set ns ""
		set i [string length $s]
		while {[incr i -1] >= 0} {
			set c [string index $s $i]
			if {[string is ascii -strict $c]} {set icount 1} else {set icount 2}
			incr slen $icount
			if {$slen > $n} {
				if {$icount == 2} {set ns " $ns"}
				return $ns
			}
			set ns $c$ns
		}
		set sep [expr {$n - $slen}]
		set i 0
		set rstr ""
		foreach c [split [string repeat $p $n] {}] {
			if {[string is ascii -strict $c]} {set icount 1} else {set icount 2}
			incr i $icount
			if {$i > $sep} {
				if {$icount == 2 && $i - $icount != $sep} {
					return [append rstr " $ns"]
				}
				return [append rstr $ns]
			}
			append rstr $c
		}
	}
	proc ltrim     {s {c " "}} {string trimleft $s $c}
	proc mid       {str pos len} {incr pos -1; string range $str $pos [expr {$pos+$len-1}]}
	
	proc repeat    {s n} {string repeat $s $n}
	proc replace   {s f t} {string map [list $f $t] $s}
	proc reverse   {s} {set i [string len $s]; set r ""; while {[incr i -1] >= 0} {append r [string index  $s $i] }; return $r}
	proc right     {s n} {string range $s end-[incr n -1] end}
	proc rpad      {s n {p " "}} {
		if {$p eq {}} {set p " "}
		set i 0
		set ret ""
		foreach c [split $s[string repeat $p $n] {}] {
			if {[string is ascii -strict $c]} {set icount 1} else {set icount 2}
			incr i $icount
			if {$i > $n} {
				if {$icount == 2 && $i - $icount < $n} {
					return "$ret "
				}
				return $ret
			}
			append ret $c
		}
	}
	proc rtrim     {s {c " "}} {string trimright $s $c}
	#    substr    Sqlite built-in
	proc space     {n} {string repeat " " $n}
	#    strftime  Sqlite built-in
	proc translate {s nl ml} {
		set max [string length $nl]
		set map [list]; 
		for {set i 0} {$i < $max} {incr i} {
			lappend map [string index $nl $i] [string index $ml $i]
		}
		string map $map $s
	}
	proc trim      {s {c " "}} {string trim $s $c}
	proc to_char {num fmt} {
		set fmt [string map {G , D . L \$} $fmt]
		if {![string is double $num]} {return "####"}
		set sign [expr {$num >= 0 ? 0 : 1}]
		set nnum [expr {abs($num)}]

		set ndig [string first . $nnum]
		if {$ndig != [string last . $nnum]} {return "####"}
		set fdig [string first . $fmt]
		if {$fdig != [string last . $fmt]} {return "####"}
		if {$fdig > -1} {set D 1} else {set D 0}
		set B 0; set FM 0; set S 0;
		foreach i {1 2 3} {
			switch -- [string index $fmt 0] {
				B {set fmt [string range $fmt 1 end]; set B 1}
				S {set fmt [string range $fmt 1 end]; set S 1}
				F {
					if {[string range $fmt 0 1] eq "FM"} {
					set fmt [string range $fmt 2 end]; set FM 1
					} else {
					return "####"
					}
				}
			}
		}
		if {[string index $fmt end] eq "S"} {
			set fmt [string range $fmt 0 end-1]; set S 2
		}

		foreach {num1 num2} [split $nnum .] break
		foreach {fmt1 fmt2} [split $fmt .] break

		if {$num1 == 0} {set num1 ""}
		set str1 ""
		set ni [expr {[string length $num1] - 1}]
		set fi [string length $fmt1]
		set zero [string first "0" $fmt1]
		if {$zero == -1} {set zero $fi}
		while {[incr fi -1] >= 0} {
			set c [string index $fmt1 $fi]
			if {$c eq "0" || $c eq "9"} {
				if {$ni >= 0} {
					set str1 [string index $num1 $ni]$str1
				} else {
					if {$fi < $zero} {
						set str1 " $str1"
					} else {
						set str1 "0$str1"
					}
				}
				incr ni -1
				continue
			}
			if {$c eq ","} {
				if {$fi < $zero && $ni < 0} {
					set str1 " $str1"
				} else {
					set str1 ",$str1"
				}
				continue
			}
			return "####"
		}
		if $FM {
			set str1 [string trimleft $str1 " "]
		}
		
		set str2 ""
		set num2 [string trimright $num2 0]
		set ni 0
		set fmax [string length $fmt2]
		set fi 0
		set zero [string last "0" $fmt2]
		while {$fi < $fmax} {
			set c [string index $fmt2 $fi]
			if {$c eq "0" || $c eq "9"} {
				set nc [string index $num2 $ni]
				if {$nc ne {}} {
					append str2 $nc
				} else {
					if {$fi <= $zero} {
						append str2 "0"
					} else {
						append str2 " "
					}
				}
				incr ni
			} else {
				return "####"
			}
			incr fi
		}
		if $FM {
			set str2 [string trimright $str2 " "]
		} else {
			set str2 [string map {" " "0"} $str2]
		}

		set str $str1
		if {$D} {append str . $str2}
		if {$B} {
			if {$str == 0} {set str [string repeat " " [string length $str]]}
		}
		if {$B == 0 && $D == 0 && $FM == 0} {
			if {[string index $str end] eq " "} {
				set str [string replace $str end end "0"]
			}
		}
		set space [string last " " $str]
		switch -exact -- $S {
			0 {
				if {$sign == 1} {set s "-"}
				if {$sign == 0 && $FM == 0} {set s " "}
				if {$sign == 0 && $FM == 1} {set s ""}
				set str [string range $str 0 $space]$s[string range $str [incr space] end]
			}
			1 {
				if {$sign == 1} {set s "-"} else {set s "+"}
				set str [string range $str 0 $space]$s[string range $str [incr space] end]
			}
			2 {
				if {$sign == 1} {set str "$str-"}
				if {$sign == 0} {set str "$str+"}
			}
		}
		return $str
	}
#        upper     Sqlite built-in

	# Math Functions
	#    abs   Sqlite built-in
	proc acos  {x} {expr {acos($x)}}
	proc asin  {x} {expr {asin($x)}}
	proc atan  {x} {expr {atan($x)}}
	proc atan2 {x y} {expr {atan2($x,$y)}}
	#    avg   Sqlite built-in
	proc ceil  {n} {expr {int(ceil($n))}}
	proc cos   {x} {expr {cos($x)}}
	proc cot   {x} {if {$x!=0} {expr {1/tan($x)}}}
	proc degrees {a} {expr {$a/3.14159265358979*180.0}}
	proc exp     {x} {expr {exp($x)}}
	proc floor   {n} {expr {int(floor($n))}}
	proc greatest args { set m [lindex $args 0]
				foreach n [lrange $args 1 end] {if {$n > $m} {set m $n}}; return $m }
	proc least    args { set m [lindex $arformat %02Xgs 0]
				foreach n [lrange $args 1 end] {if {$n < $m} {set m $n}}; return $m }
	proc log     {x} {if {$x != 0} {expr {log($x)}}}
	proc log10   {x} {if {$x != 0} {expr {log10($x)}}}
	proc mod     {m n} {expr {$m % $n}}
	proc pi      args {return 3.14159265358979}
	proc pow     {x y} {expr {pow($x,$y)}}
	proc radians {x} {expr {$x*3.14159265358979/180}}
	proc rand    {} {expr {rand()}}
	#    random  Sqlite built-in
	#    round   Sqlite built-in
	proc sign    {x} {expr {$x == 0 ? 0 : ($x > 0 ? 1 : -1)}}
	proc sin     {x} {expr {sin($x)}}
	proc sqrt    {x} {expr {sqrt($x)}}
	proc tan     {x} {expr {tan($x)}}
	proc trunc   {n {d 0}} {set t [expr {pow(10,$d)}]; 
						expr {$d <= 0 ? int(int($n*$t)/$t) : int($n*$t)/$t}}
	
	# DateTime Functions
	proc now {} {clock format [clock seconds] -format "%Y-%m-%d %H:%M:%S" -gmt 0};#Access now()
#    julianday Sqlite built-in
#    datetime  Sqlite built-in
#    date      Sqlite built-in
#    strftime  Sqlite built-in
	
	# Other Functions
#    min         Sqlite built-in
#    max         Sqlite built-in
#    coalesce    Sqlite built-in
#    nullif      Sqlite built-in
#    ifnull      Sqlite built-in
#    last_insert_rowid Sqlite built-in
	proc user {} {set env(USERNAME)}

	# regexp, regsub
	interp alias {} [namespace current]::regexp {} regexp
	interp alias {} [namespace current]::regsub {} regsub
	

	# md5, md5_hmac
	if {![catch {package require md5}]} {
		proc md5 {data {binary 0}} {
			set data [_getEncodedByteArray $data $binary]
			return [::md5::md5 -hex $data]
		}
		proc md5_hmac {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set data [_getEncodedByteArray $data $binary]
			return [::md5::hmac -hex -key $key $data]
		}
	}
	# md5crypt, apr_crypt
	if {![catch {package require md5crypt}]} {
		proc md5_crypt {pass salt {binary 0}} {
			set pass [_getEncodedByteArray $pass 0]
			set salt [_getEncodedByteArray $salt $binary]
			return [::md5crypt::md5crypt $pass $salt]
		}
		proc apr_crypt {pass salt {binary 0}} {
			set pass [_getEncodedByteArray $pass 0]
			set salt [_getEncodedByteArray $salt $binary]
			return [::md5crypt::aprcrypt $pass $salt]
		}
	}
	# sha1, sha1_hmac
	if {![catch {package require sha1}]} {
		proc sha1 {data {binary 0}} {
			set data [_getEncodedByteArray $data $binary]
			return [::sha1::sha1 -hex $data]
		}
		proc sha1_hmac {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set data [_getEncodedByteArray $data $binary]
			return [::sha1::hmac -hex -hey $key $data]
		}
	}
	# aes_encrypt, aes_decrypt
	if {![catch {package require aes}]} {
		proc aes_encrypt {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set data [_getEncodedByteArray $data $binary]
			return [::aes::aes -mode ecb -dir encrypt -key $key -- $data]
		}
		proc aes_decrypt {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set result [::aes::aes -mode ecb -dir decrypt -key $key -- $data]
			set result [_getDataFromByteArray $result $binary]
			return $result
		}
	}
	# blowfish_encrypt, blowfish_decrypt
	if {![catch {package require blowfish}]} {
		proc blowfish_encrypt {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set data [_getEncodedByteArray $data $binary]
			return [::blowfish::blowfish -mode ecb -dir encrypt -key $key -- $data]
		}
		proc blowfish_decrypt {key data {binary 0}} {
			set key  [_getEncodedByteArray $key 0]
			set result [::blowfish::blowfish -mode ecb -dir decrypt -key $key -- $data]
			set result [_getDataFromByteArray $result $binary]
			return $result
		}
	}
	# des_decrypt, des_encrypt
	if {![catch {package require des}]} {
		proc des_encrypt {key data} {
			set key  [_getEncodedByteArray $key 0]
			set data [_getEncodedByteArray $data $binary]
			return [::DES::des -mode ecb -dir encrypt -key $key -- $data]
		}
		proc des_decrypt {key data} {
			set key  [_getEncodedByteArray $key 0]
			set result [::DES::des -mode ecb -dir decrypt -key $key -- $data]
			set result [_getDataFromByteArray $result $binary]
			return $result
		}
	}
	# base64_encode, base64_decode
	if {![catch {package require Trf}]} {
		proc base64_encode {data {binary 0}} {
			set data [_getEncodedByteArray $data $binary]
			return [::base64 -mode encode -- $data]
		}
		proc base64_decode {data {binary 0}} {
			set result [::base64 -mode decode -- $data]
			set result [_getDataFromByteArray $result $binary]
			return $result
		}
	}
	# uuid
	if {![catch {package require uuid}]} {
		proc uuid {} {return [::uuid::uuid generate]}
	}
	# compress
	if {![catch {package require Trf}]} {
		proc compress {data {binary 0}} {
			set data [_getEncodedByteArray $data $binary]
			return [zip -mode compress -- $data]
		}
		proc decompress {data {binary 0}} {
			set result [zip -mode decompress -- $data]
			set result [_getDataFromByteArray $result $binary]
			return $result
		}
	}
	# file io
	if {![catch {package require Trf}]} {
		proc write_file {file val {binary 0}} {
			variable version
			set file [_getTclString $file]
			if {[catch {open $file w} fp]} {error "failed to open '$file'"}
			set val [_getEncodedByteArray $val $binary]
			if {$binary && $version == 2} {
				set val [::base64 -mode decode $val]
			}
			fconfigure $fp -translation binary -encoding binary
			puts -nonewline $fp $val
			close $fp
			return
		}
		proc read_file {file {binary 0}} {
			variable version
			set file [_getTclString $file]
			if {[catch {open $file r} fp]} {error "failed to open '$file'"}
			fconfigure $fp -translation binary -encoding binary
			set val [::read $fp]
			close $fp
			set val [_getDataFromByteArray $val $binary]
			if {$binary && $version == 2} {
				set val [::base64 -mode encode $val]
			}
			return $val
		}
	}
}



proc Sqlite::defineFuctions {interp} {
	if {$interp eq {}} {return {}}

	set ns ::Sqlite::func

	foreach cmd [info command ${ns}::*] {
		$interp alias $cmd $cmd
	}
}

proc Sqlite::installFunctions {interp} {
	$interp eval ::Sqlite::func::install db
}

proc Sqlite::loadExtension {interp} {
	variable info
	set version [Sqlite::getCurrentFullVersion]
	set enableLoadExtension [expr {[::Util::versionCompare $version 3.3.6] > 0}]
	foreach {enable path entrypoint} $info(extension) {
		set path [file normalize $path]
		if {[string is boolean $enable] && !$enable} {continue}
		if {![file exists $path] || ![file isfile $path]} {
			::Message::show -type apperror \
			-message [msgcat::mc {Failed to open the file.}] \
			-detail $path
			continue
		}
		set ext [string tolower [file extension $path]]
		if {$ext eq ".tcl"} {
			if {[catch {$interp eval source "{$path}"} msg]} {
				::Message::show -type apperror \
				-message "error in sourcing the script." \
				-detail $path\n$msg
			}
			continue
		}
		if {[lsearch {.dll .so .dylib} $ext] != -1 && $enableLoadExtension} {
			if {$entrypoint eq ""} {
				set query "SELECT load_extension('$path');"
			} else {
				set query "SELECT load_extension('$path', '$entrypoint');"
			}
			if {[catch {db eval $query} msg]} {
				::Message::show -type apperror \
				-message "error in loading the library." \
				-detail $path\n$msg
			}
			continue
		}
		::Message::show -type apperror \
		-message [msgcat::mc {Unknown extension type.}] \
		-detail $path\n($ext)
	}
}

proc Sqlite::init {} {
	variable data {}
	variable status
	variable interp
	set status(haveSqlite2) 0
	set status(haveSqlite3) 0
	set interp(2) [interp create interp2]
	set interp(3) [interp create interp3]
	$interp(2) eval [list set auto_path $::auto_path]
	$interp(3) eval [list set auto_path $::auto_path]
	set ilist {}

	if {![catch {$interp(2) eval "package require sqlite 2.0"}]} {
		if {[$interp(2) eval "sqlite -encoding"] eq "iso8859"} {
		defineFuctions $interp(2)
		lappend ilist 2
		} else {
		::Message::show -type apperror -textbox 0 \
		-message [msgcat::mc "Unsupported tclsqlite. I require a tclsqlite compiled for ISO8859 encoding."]
		}
	}

	if {![catch {$interp(3) eval "package require sqlite3"}]} {
		$interp(3) eval "sqlite3 db :memory:"
		set enc [$interp(3) eval "db eval {PRAGMA encoding}"]
		$interp(3) eval "db close"
		if {$enc eq "UTF-8"} {
		defineFuctions $interp(3)
		lappend ilist 3
		} else {
		::Message::show -type apperror -textbox 0 \
		-message [msgcat::mc "Unsupported SQLite3"]
		}
	}
	
	foreach n $ilist {
	set status(haveSqlite$n) 1
	$interp($n) eval "set version $n"
	}
}
proc Sqlite::dump {} {
	variable info
	set session "#::[info level [info level]]\n"
	::append session "array set [namespace current]::info \[list [array get info]\]\n"
	return $session
}
Sqlite::init
;#>>>


# Tool namespace is for nonmodal dialog
namespace eval Tool::Searcher {;#<<<
# Table Searcher
	variable root .cmdSearcher
	variable info
	array set info {
		historyMax 30
		query_s {}
		query_r {}
		history_s {}
		history_r {}
		pattern none
		nocase 1
	}
}

proc Tool::Searcher::dump {} {
	variable info
	return "#Tool::Searcher::dump\narray set [namespace current]::info \[list [array get info]\]\n"
}

proc Tool::Searcher::run {} {
	variable info
	variable root
	if {![winfo exists $root]} {
		set root [toplevel $root]
		wm withdraw $root
		wm title $root [msgcat::mc Search]
		set cancelAction "wm withdraw $root"
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		bind $root <Escape> $cancelAction
		bind $root <F3> $cancelAction

		ttk::frame $root.f -padding 4
		ttk::frame $root.f.cmd  -padding 4
		ttk::frame $root.f.detail -padding 4
		pack $root.f -expand yes -fill both
		pack $root.f.cmd    -side right -fill both
		pack $root.f.detail -side top   -fill both
		
		# Buttons
		ttk::button $root.f.cmd.searchUp -text [msgcat::mc "Search Up"]
		ttk::button $root.f.cmd.searchDn -text [msgcat::mc "Search Down"]
		ttk::button $root.f.cmd.replace  -text [msgcat::mc Replace]
		ttk::button $root.f.cmd.close    -text [msgcat::mc Close] -command $cancelAction
		foreach w [winfo children $root.f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $root.f.cmd.searchUp $root.f.cmd.searchDn $root.f.cmd.replace -side top -fill x -pady 2
		pack $root.f.cmd.close -side bottom -fill x -pady 2

		$root.f.cmd.searchUp configure -command "[namespace current]::search up"
		$root.f.cmd.searchDn configure -command "[namespace current]::search down"
		$root.f.cmd.replace  configure -command "[namespace current]::replace"
		
		# Details
		ttk::label       $root.f.detail.slabel -text [msgcat::mc Search]
		ttk::label       $root.f.detail.rlabel -text [msgcat::mc Replace]
		ttk::combobox    $root.f.detail.squery
		ttk::combobox    $root.f.detail.rquery
		Util::bindEditorPopCommon $root.f.detail.squery
		Util::bindEditorPopCommon $root.f.detail.rquery
		bind $root.f.detail.squery <Shift-Return> "[namespace current]::search up"
		bind $root.f.detail.squery <Return> "[namespace current]::search down"
		
		grid $root.f.detail.slabel  -row 0 -column 0 -sticky w
		grid $root.f.detail.squery  -row 0 -column 1 -sticky ew -pady 2
		grid $root.f.detail.rlabel  -row 1 -column 0 -sticky w
		grid $root.f.detail.rquery  -row 1 -column 1 -sticky ew -pady 2
		grid columnconfigure $root.f.detail 1 -weight 1

		foreach c {s r} {	
		$root.f.detail.${c}query configure \
		-textvariable [namespace current]::info(query_${c}) \
		-postcommand "$root.f.detail.${c}query configure -values \
			\$[namespace current]::info(history_${c})
		set _idx [lsearch -exact \$[namespace current]::info(history_${c})]
		if {\$_idx >= 0} { $root.f.detail.${c}query current \$_idx }" 
		}

		set of [ttk::labelframe $root.f.detail.option -text [msgcat::mc "Pattern Match"]]
		set rval [namespace current]::info(pattern)
		set nval [namespace current]::info(nocase)
		ttk::radiobutton $of.rexact -variable $rval -value none -text [msgcat::mc None]
		ttk::radiobutton $of.rgrob  -variable $rval -value glob -text Glob
		ttk::radiobutton $of.rregex -variable $rval -value regexp -text [msgcat::mc RegExp]
		ttk::checkbutton $of.nocase -variable $nval -text [msgcat::mc NoCase]
		Util::bindMouseEvent $of.rexact 
		Util::bindMouseEvent $of.rgrob  
		Util::bindMouseEvent $of.rregex 
		Util::bindMouseEvent $of.nocase
			grid $of.rexact -row 0 -column 0 -sticky ew
			grid $of.rgrob  -row 0 -column 1 -sticky ew
			grid $of.rregex -row 0 -column 2 -sticky ew
			grid $of.nocase -row 1 -column 0 -columnspan 3 -sticky ew
			grid columnconfigure $of 0 -weight 1
			grid columnconfigure $of 1 -weight 1
			grid columnconfigure $of 2 -weight 1

		grid $of -row 2 -column 0 -columnspan 3 -sticky news -pady 4
 		
		::Util::updateWindowSize $root
		wm transient $root .
		wm resizable $root 0 0
		Util::moveCenter $root
	}
	wm deiconify $root
	raise $root
	focus -force $root.f.detail.squery
}

proc Tool::Searcher::replace {} {
	variable info
	dputs {replace}

	# save query
	_saveQueryS
	_saveQueryR

	# get current index
	set index [lindex [::Table::getActive] 0]
	if {$index eq ""} {return}
	if {[scan $index %d,%d row col] != 2} return

	if {$info(pattern) eq "none"} {
		if {$info(nocase) == 0} {
			Table::updateRow $index \
			[string map [list $info(query_s) $info(query_r)] $::Table::table($index)]
		} else {
			Table::updateRow $index \
			[string map -nocase [list $info(query_s) $info(query_r)] $::Table::table($index)]
		}
	}
	if {$info(pattern) eq "glob"} {
		if {$info(nocase) == 0} {
			if {[string match $info(query_s) $::Table::table($index)]} {
			Table::updateRow $index $info(query_r)
			}
		} else {
			if {[string match -nocase $info(query_s) $::Table::table($index)]} {
			Table::updateRow $index $info(query_r)
			}
		}
	}
	if {$info(pattern) eq "regexp"} {
		if {$info(nocase) == 0} {
			set op --
		} else {
			set op -nocase
		}
		set r [regsub $op -- $info(query_s) $::Table::table($row,$col) $info(query_r)]
		Table::updateRow $index $r
	}
	Table::see $index
}

proc Tool::Searcher::search {dir} {
	variable info
	dputs {search $dir}

	# save query
	_saveQueryS
	
	# get current index
	set index [lindex [::Table::getActive] 0]
	if {$index eq ""} {return}
	if {[scan $index %d,%d row col] != 2} return
	
	set max  [Table::getRow]
	if {$max <= 1} return
	
	if {$dir eq "up"} {
		set inum -1
	} else {
		set inum 1
	}
	incr row $inum
	
	if {$info(pattern) eq "none" || $info(pattern) eq "glob"} {
		switch -exact -- $info(pattern) {
			"none" {set pattern *[string map {* \\* [ \\[ ] \\] ? \\? \\ \\\\} $info(query_s)]*}
			"glob" {set pattern $info(query_s)}
		}
		# search
		if {$info(nocase) == 0} {
			while {$row > 0 && $row < $max} {
				if {[string match $pattern \
					$::Table::table($row,$col)]} {
					Table::see $row,$col; return
				}
				incr row $inum
			}
		} else {
			while {$row > 0 && $row < $max} {
				if {[string match -nocase $pattern \
					$::Table::table($row,$col)]} {
					Table::see $row,$col; return
				}
				incr row $inum
			}
		}
	}
	if {$info(pattern) eq "regexp"} {
		if {$info(nocase) == 0} {
			set op --
		} else {
			set op -nocase
		}
		while {$row > 0 && $row < $max} {
			set r [regexp -inline $op  -- $info(query_s) $::Table::table($row,$col)]
			if {[llength $r] > 0} {
				Table::see $row,$col
				return
			}
			incr row $inum
		}
	}
	# Not found
	dputs {not found}
	Table::see $index
	return
}

proc Tool::Searcher::_saveQueryS {} {
	variable info
	set max [expr {$info(historyMax) - 1}]
	set idx [lsearch -exact $info(history_s) $info(query_s)]
	if {$idx < 0} {
		set info(history_s) [concat $info(query_s) $info(history_s)]
		set info(history_s) [lrange $info(history_s) 0 $max]
	}
}

proc Tool::Searcher::_saveQueryR {} {
	variable info
	set max [expr {$info(historyMax) - 1}]
	set idx [lsearch -exact $info(history_r) $info(query_r)]
	if {$idx < 0} {
		set info(history_r) [concat $info(query_r) $info(history_r)]
		set info(history_r) [lrange $info(history_r) 0 $max]
	}
}
;#>>>

# Data Editor
namespace eval Tool::DataEditor {;#<<<
	variable root .cmdDataEditor
	variable text
	array set tab {}
	variable note {}

	# var
	variable info
	array set info {
		currenttab 0
		inisize   {290 240}
		minsize   {200 150}

		rawdata   {}

		db.table  {}
		rowid     {}
		rowseq    {}
		colname   {}
		tactive   {}
		editable  0

		imgdir    {}
		img       {::image::DataEditorCanvasImage}
		drag      0
		badimage  1
	}
	variable state ""
}

proc Tool::DataEditor::run {} {;#<<<
	variable info
	variable root
	variable note
	variable tab
	variable state

	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	
	# make gui
	_init
	
	# update data of DataEditor
	set state show
	::Table::updateDataEditor
	::Table::see

	wm transient $root .
	wm deiconify $root
	switch -exact -- [$note index current] {
		0 {focus -force $tab(text).f.text}
		1 {focus -force $tab(image).f.canvas}
		default {focus -force $root}
	}
	raise $root
	return
};#>>>

proc Tool::DataEditor::_init {} {;#<<<
	variable tab
	variable note
	variable text
	variable info
	variable root

	if {![winfo exists $root]} {
		if {[catch {package require Img}]} {
			set ::pref(openTypeImage) $::pref(openTypeImageNoImg)
		} else {
			set ::pref(openTypeImage) $::pref(openTypeImageWithImg)
		}

		set root [toplevel $root]
		wm withdraw $root
		set f [ttk::frame $root.f -padding 3]
		pack $f -expand yes -fill both
		wm title $root [msgcat::mc "Data Editor"]

		# Command Buttons
		ttk::frame $f.cmd
		ttk::button $f.cmd.update  -text [msgcat::mc Update]
		ttk::button $f.cmd.close   -text [msgcat::mc Close]
		ttk::button $f.cmd.up    -image $::img(arrowUp)
		ttk::button $f.cmd.down  -image $::img(arrowDown)
		ttk::button $f.cmd.left  -image $::img(arrowLeft)
		ttk::button $f.cmd.right -image $::img(arrowRight)
		foreach w [winfo children $f.cmd] {
			Util::bindMouseEvent $w
		}
		foreach w [list $f.cmd.up $f.cmd.down $f.cmd.left $f.cmd.right] {
			$w configure -style Toolbutton -pad 4 -compound image
		}
		pack $f.cmd -fill x -side bottom -pady 4
		pack $f.cmd.close $f.cmd.update -side right -padx 2
		pack $f.cmd.up $f.cmd.down $f.cmd.left $f.cmd.right \
		-side left -padx 0
		
		# Root binding
		set cancelAction [namespace code {wm withdraw $root;set state hide}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$f.cmd.update configure -command [namespace code {updateDatabase}]
        $f.cmd.close  configure -command $cancelAction
		$f.cmd.up    configure -command {::Table::Navi $::Table::tablew up;    break}
		$f.cmd.down  configure -command {::Table::Navi $::Table::tablew down;  break}
		$f.cmd.left  configure -command {::Table::Navi $::Table::tablew left;  break}
		$f.cmd.right configure -command {::Table::Navi $::Table::tablew right; break}
		bind $root <F4> $cancelAction

		# Notebook root
		set note [ttk::notebook $f.notebook -width 10]
		pack $f.notebook -fill both -expand yes
		$f.notebook add [ttk::frame $f.notebook.text]  -sticky news -text [msgcat::mc "Text"]
		$f.notebook add [ttk::frame $f.notebook.image] -sticky news -text [msgcat::mc "Image"]
		set tab(text) $f.notebook.text
		set tab(image) $f.notebook.image
		bind $f.notebook <<NotebookTabChanged>> +[namespace code {changeTab}]
		# Text
		#-------------------------------------------------
		ttk::frame $tab(text).f -style TEntry -border 2
		text $tab(text).f.text \
			-highlightthickness 0 \
			-height 0 \
			-relief flat \
			-yscrollcommand "$tab(text).f.y set"
		ttk::scrollbar $tab(text).f.y -takefocus 0 -orient v -command "$tab(text).f.text yview"
		grid $tab(text).f.text -row 0 -column 0 -sticky news
		grid $tab(text).f.y    -row 0 -column 1 -sticky ns
		grid rowconfigure    $tab(text).f 0 -weight 1
		grid columnconfigure $tab(text).f 0 -weight 1
		pack $tab(text).f -fill both -expand yes
		set m [Util::makeEditorPop $tab(text).f]
		$m add separator
		$m add command -label [msgcat::mc Update] -accelerator $::pref(shortmod)+Return

		set text $tab(text).f.text
		bind $text <<B3>> "if {\[%W cget -state\] ne {disabled}} {
					focus %W; tk_popup $m %X %Y}; break;"
		bind $text <$::pref(modifier)-Return> \
			[namespace code {updateDatabase; break;}]
		# add and init vi bind for Tool::DataEditor
		trace add variable ::Table::info(vibind) write [namespace current]::useViBind
		useViBind ::Table::info vibind ""

		Cmd::bindTextFocusTtkEntry $tab(text).f.text $tab(text).f
		# Image
		#-------------------------------------------------
		ttk::frame $tab(image).f
		switch -exact -- $::tile_currentTheme {
			xpnative { $tab(image).f configure -relief flat -border 2}
			default { $tab(image).f configure -relief sunken -border 2 }
		}
		canvas $tab(image).f.canvas \
			-cursor fleur \
			-relief flat \
			-highlightthickness 0 \
			-borderwidth 0 \
			-width 0 \
			-height 0 \
			-background black \
			-confine 1 \
			-xscrollcommand "$tab(image).f.x set" \
			-yscrollcommand "$tab(image).f.y set"
		ttk::scrollbar $tab(image).f.x -takefocus 0 -orient h -command "$tab(image).f.canvas xview"
		ttk::scrollbar $tab(image).f.y -takefocus 0 -orient v -command "$tab(image).f.canvas yview"
		$tab(image).f.canvas create image 0 0 -anchor nw -tags image
		$tab(image).f.canvas create text 0 0 -anchor nw -fill white -tags text
		
		grid $tab(image).f.canvas -row 0 -column 0 -sticky news
		grid $tab(image).f.y      -row 0 -column 1 -sticky ns
		grid $tab(image).f.x      -row 1 -column 0 -sticky ew
		grid rowconfigure    $tab(image).f 0 -weight 1
		grid columnconfigure $tab(image).f 0 -weight 1
		pack $tab(image).f -side left -fill both -expand yes

		bind $tab(image).f.canvas <ButtonPress-1> \
			[namespace code {set info(drag) 1; %W scan mark %x %y}]
		bind $tab(image).f.canvas <Motion> \
			[namespace code {if {$info(drag) && !$info(badimage)} {%W scan dragto %x %y 1}}]
		bind $tab(image).f.canvas <ButtonRelease-1> \
			[namespace code {set info(drag) 0}]

		Util::bindMouseEvent [ttk::button $tab(image).load -text [msgcat::mc Load]]
		Util::bindMouseEvent [ttk::button $tab(image).save -text [msgcat::mc Save]]
		ttk::label $tab(image).info -anchor w -padding 2 -text "\nw:\nh:"

		pack $tab(image).load $tab(image).save -side top -padx 2 -pady 2
		pack $tab(image).info -side top -padx 2 -pady 2 -fill x
		$tab(image).load configure -command [namespace current]::loadImage
		$tab(image).save configure -command [namespace current]::saveImage
		
		updateTextStyle

		# set size
		wm transient $root .
		eval [subst {wm minsize $root $info(minsize)}]
		::Util::updateWindowSize $root [join $info(inisize) x]
		::Util::moveCenter $root $info(inisize)
;#        wm resizable $root 0 0
		setInfo \
			{} \
			{} \
			{} \
			{} \
			0
	}
	set info(currenttab) [$note index current]
};#>>>

proc Tool::DataEditor::useViBind {n1 n2 op} {
	variable text
	# n1(n2) = Table::info(vibind) maybe
	if [set ${n1}($n2)] {
		bind $text <Control-h> {::Table::Navi $::Table::tablew left;  break}
		bind $text <Control-j> {::Table::Navi $::Table::tablew down;  break}
		bind $text <Control-k> {::Table::Navi $::Table::tablew up;    break}
		bind $text <Control-l> {::Table::Navi $::Table::tablew right; break}
	} else {
		bind $text <Control-h> {break}
		bind $text <Control-j> {break}
		bind $text <Control-k> {break}
		bind $text <Control-l> {break}
	}
}

proc Tool::DataEditor::changeTab {} {
	variable info
	variable note
	variable tab
	
	set oldtab $info(currenttab) 
	set newtab [$note index current]
	updateWidget  $newtab
	set info(currenttab) $newtab
}

proc Tool::DataEditor::getStatus {} {
	variable state
	return $state
}

proc Tool::DataEditor::setInfo {dbtable rowid tactive colname editable} {
	variable root
	variable tab
	variable info

	set info(db.table) $dbtable
	set info(rowid)    $rowid
	set info(colname)  $colname
	set info(tactive)  $tactive
	set info(editable) $editable
	
	if {$editable} {
		if {[$root.f.cmd.update cget -state] ne "normal"} {
			$root.f.cmd.update configure -state normal
			$tab(text).f.pop entryconfigure [msgcat::mc Update] -state normal
		}
	} else {
		if {[$root.f.cmd.update cget -state] ne "disabled"} {
			$root.f.cmd.update configure -state disabled
			$tab(text).f.pop entryconfigure [msgcat::mc Update] -state disabled
		}
	}
}

# set rowdata to DataEditor
proc Tool::DataEditor::setRawData {rawdata} {
	variable info
	variable note
	set info(rawdata) $rawdata
	updateWidget [$note index current]
}

# update db and Table from DataEditor.
proc Tool::DataEditor::updateDatabase {} {
	variable info
	variable note

	if {!$info(editable)} {return}

	updateRawData [$note index current]
	set data $info(rawdata)
	
	set data $info(rawdata)
	set blob 0
	switch -exact -- [$note index current] {
		0 { set blob 0; Cmd::encodingConvert from data }
		1 { set blob 1 }
		default {set blob 0}
	}
	Table::updateRow $info(tactive) $data $blob
}

# set rawdata from widget
proc Tool::DataEditor::updateRawData {tabnum} {
	variable info
	variable tab
	switch -exact -- $tabnum {
		0 {
			#Text
			set data [$tab(text).f.text get 1.0 {end -1 chars}]
			Cmd::encodingConvert to data
			set info(rawdata) $data
		}
		1 {
			#Image
			# Do nothing
		}
	}
}

# set widget from rawdata
proc Tool::DataEditor::updateWidget {tabnum} {
	variable info
	variable tab

	switch -exact -- $tabnum {
		0 { 
			# Text
			set data $info(rawdata)
			Cmd::encodingConvert from data
			$tab(text).f.text edit reset
			$tab(text).f.text delete 1.0 end
			$tab(text).f.text insert 1.0 $data
		}
		1 { 
			# Image
			set data $info(rawdata)
			$tab(image).f.canvas xview moveto 0
			$tab(image).f.canvas yview moveto 0
			catch {image delete $info(img)}
			if {[catch {image create photo $info(img) -data $data}]} {
				$tab(image).f.canvas itemconfigure image -image {}
				$tab(image).f.canvas itemconfigure text  -text [msgcat::mc "Unknown Image Format"]
				$tab(image).f.canvas configure \
					-scrollregion [$tab(image).f.canvas bbox text]
				set info(badimage) 1
				$tab(image).info configure -text "\nw:\nh:"
				
			} else {
				$tab(image).f.canvas itemconfigure image -image $info(img)
				$tab(image).f.canvas itemconfigure text  -text {}
				$tab(image).f.canvas configure \
					-scrollregion [$tab(image).f.canvas bbox image]
				set info(badimage) 0
				if {[catch {::Util::getImageFormat data} ret]} {
					set fname ""
					set w ""
					set h ""
				} else {
					set fname [string toupper $ret]
					set w [image width $info(img)]
					set h [image height $info(img)]
				}
				$tab(image).info configure -text "$fname\nw:$w\nh:$h"
			}
		}
	}
}

proc Tool::DataEditor::loadImage {} {
	variable root
	variable info
	variable note

	set file [tk_getOpenFile -parent $root -initialdir $info(imgdir) -filetypes $::pref(openTypeImage)]
	if {$file eq ""} return
	if {![file exists $file]} return
	
	set info(imgdir) [file dir $file]
	set fp [open $file r]
	fconfigure $fp -translation binary -encoding binary
	set data [read $fp]
	close $fp

	# We should memorize used filetype, and use it as a default filetype in next time.
	# But tk_getOpenFile can't specify defualt filetype.
	# TODO: Is there any good idea?

	if {[Sqlite::getCurrentVersion] == 2} {
		setRawData [::base64 -mode encode -- $data]
	} else {
		setRawData $data
	}
}

proc Tool::DataEditor::saveImage {} {
	variable root
	variable info
	variable note
	variable tab

	set file [tk_getSaveFile -parent $root -initialdir $info(imgdir)]
	if {$file eq ""} return
	set info(imgdir) [file dir $file]
	
	updateRawData [$note index current]
	
	if {[Sqlite::getCurrentVersion] == 2} {
		set data [::base64 -mode decode -- $info(rawdata)]
	} else {
		set data $info(rawdata)
	}
	set fp [open $file w]
	fconfigure $fp -translation binary -encoding binary
	puts -nonewline $fp $data
	close $fp
}

proc Tool::DataEditor::clear {} {
	variable tab
	variable info
	$tab(text).f.text edit reset
	$tab(text).f.text delete 1.0 end
	set info(rawdata)   {}
}

proc Tool::DataEditor::reset {} {
	variable info
	set info(rawdata)   {}
	set info(dbname)    {}
	set info(tablename) {}
	set info(rowid)     {}
	set info(colname)   {}
	clear
}

proc Tool::DataEditor::updateTextStyle {} {
	variable root
	variable tab
	if {[winfo exists $root]} {
		set n [expr {72.0 / [winfo pixels . 1i]}]
		upvar ::SQLText::info info
		$tab(text).f.text configure \
			-font $info(font) \
			-tabs [expr {$n * [font measure $info(font) 0] * $info(tab)}]
	}
}
;#>>>

# GUICmd namespace is for modal dialog command
namespace eval GUICmd {};#<<<

# Create / Modyfy Table Schema
namespace eval GUICmd::TableBuilder {;#<<<
	# widgets
	variable root ""
	variable tree ""
	variable columnDialog ""
	# dialog status tkwait
	variable status cancel
	variable columnDialogStatus cancel
	# vars
	array set info {
		iniSize {560 320}
		minSize {560 320}
		runmode   create
		showonly  0
		dbname    ""
		tablename ""
		oldtabledata {}
	}
	variable conflicttype [list ROLLBACK ABORT FAIL IGNORE REPLACE]
	array set tableconst {check "" checkconflict ""}
}

# - run create   Create New Table
# - run modify   Modify Table Schema
proc GUICmd::TableBuilder::run {cmd} {;#<<<
	variable root
	variable state
	variable info

	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}

	switch -- $cmd {
		modify  {set info(runmode) modify}
		create  -
		default {set info(runmode) create}
	}
	
	if {$cmd eq "modify"} {
		set _tableData [list]
		set type ""
		foreach {_tableData} [::Dbtree::getSelection] {
			if {[lindex $_tableData 2] eq {}} continue
			set db    [lindex $_tableData 2 0];# db name
			set type  [lindex $_tableData 2 1];# table/view
			set table [lindex $_tableData 2 2];# table name
			if {$type eq "table"} break
		}
		if {$type ne "table"}  {
			::Message::show -type appinfo -textbox 0 \
			-message [msgcat::mc "Please select a table."]
			return
		}
		set info(dbname)    $db
		set info(tablename) $table
		set sql [lindex $_tableData 2 5]
		set tabledata [SQLParser::parseCreateTable [SQLParser::removeComment $sql]]
		if {$tabledata eq ""} {
			::Message::show -type apperror -textbox 0 \
			-message [msgcat::mc "Failed to parse SQL. Unsupported table schema."]
			return
		}
		_init $tabledata
	} else {
		_init
	}
	
	wm transient $root .
	wm deiconify $root
	::tk::SetFocusGrab $root $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw

	if {$state eq "cancel"} return
};#>>>

proc GUICmd::TableBuilder::_init {{tabledata {}}} {;#<<<
	variable root
	variable tree
	# vars
	variable info
	variable conflicttype
	variable tableconst

	if {$root eq ""} {
		set cancelAction [namespace code {
			wm withdraw $root; set state cancel}]
		set root [toplevel .cmdTableBuilder]
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		bind $root <Escape> $cancelAction
		wm withdraw $root

		ttk::frame $root.f -padding 4
		pack $root.f -expand yes -fill both
		
		# DB.Table Names
		ttk::frame $root.f.table -padding 4
		ttk::label $root.f.table.dbl -text "[msgcat::mc Database] : "
		ttk::combobox $root.f.table.db -width 16 -state readonly \
			-textvariable [namespace current]::info(dbname)
		ttk::label $root.f.table.tablel -text "[msgcat::mc Table] : "
		ttk::entry $root.f.table.table \
			-textvariable  [namespace current]::info(tablename)
		Util::bindEditorPopCommon $root.f.table.table
		pack $root.f.table.dbl $root.f.table.db \
		$root.f.table.tablel $root.f.table.table\
		-side left
		pack $root.f.table -side top -fill x

		# Tree List of Columns
		ttk::frame $root.f.cmd -padding 4
		pack $root.f.cmd -side top -fill both -expand yes

		ttk::frame $root.f.cmd.treef
		pack [Cmd::createTreectrl $root.f.cmd.treef both] -side left -fill both -expand yes
		set tree $root.f.cmd.treef.tlist

		ttk::button $root.f.cmd.new    -text [msgcat::mc "Add Column"] \
			-command [namespace code {addColumn}]
		ttk::button $root.f.cmd.delete -text [msgcat::mc "Delete Column"] \
			-command [namespace code {deleteColumn}]
		ttk::button $root.f.cmd.modify -text [msgcat::mc "Modify Column"] \
			-command [namespace code {modifyColumn}]
		ttk::button $root.f.cmd.up     -text [msgcat::mc Up] \
			-command [namespace code {moveColumn up}]
		ttk::button $root.f.cmd.down   -text [msgcat::mc Down] \
			-command [namespace code {moveColumn down}]
		foreach w [winfo children $root.f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $root.f.cmd.new $root.f.cmd.delete \
				$root.f.cmd.modify $root.f.cmd.up \
				$root.f.cmd.down -side top -fill x -padx {8 0}

		# Table Constraint
		ttk::labelframe $root.f.const -padding 4 -text [msgcat::mc Table-Constraints]
		ttk::label $root.f.const.checkl -text "Check : "
		ttk::entry $root.f.const.check \
			-textvariable [namespace current]::tableconst(check)
		ttk::combobox $root.f.const.checkconflict \
			-textvariable [namespace current]::tableconst(checkconflict) \
			-state read -values [concat {{}} $conflicttype]
		Util::bindEditorPopCommon $root.f.const.check

		pack $root.f.const -fill x -side top -padx 4
		pack $root.f.const.checkl   -side left
		pack $root.f.const.check    -side left -fill x -expand yes
		pack $root.f.const.checkconflict -side left -padx {2 0}

		trace add variable [namespace current]::tableconst(check)\
			write [namespace current]::tableconstCheckEvent
		proc tableconstCheckEvent args [subst {
			set cvalue \[set [namespace current]::tableconst(check)\]
			set cvalue \[string trim \$cvalue\]
			if {\[string length \$cvalue\] > 0} {
				set state "readonly"
			} else {
				set state "disabled"
			}
			$root.f.const.checkconflict configure -state \$state
		}]

		# Ok, Cancel Buttons
		ttk::frame $root.f.cmd2
		ttk::checkbutton $root.f.cmd2.showonly \
			-text [msgcat::mc "Show Only"] \
			-variable [namespace current]::info(showonly)
		ttk::button $root.f.cmd2.ok -text OK \
			-command [namespace code {
				if {[runSQL]==0} {
					wm withdraw $root
					set state ok
				}
			}]
		ttk::button $root.f.cmd2.cancel -text Cancel\
			-command $cancelAction

		foreach w [winfo children $root.f.cmd2] {
			Util::bindMouseEvent $w
		}
		pack $root.f.cmd2.cancel $root.f.cmd2.ok \
		$root.f.cmd2.showonly -side right -padx 2
		pack $root.f.cmd2 -side bottom -fill x -pady {8 0}

		# This was really for Debug usage. but it is useful.
		ttk::button $root.f.cmd2.dumpsql -text [msgcat::mc "Confirm Query"] \
		-command [namespace code {
			if {$info(runmode) eq "create"} {
				::Message::show -type appinfo \
				-message "SQL Query" \
				-detail [buildCreateQuery $info(dbname) $info(tablename)] \
				-parent $root
			} else {
				::Message::show -type appinfo \
				-message "SQL Query" \
				-detail [buildModifyQuery $info(dbname) $info(tablename)] \
				-parent $root
			}
		}]
		Util::bindMouseEvent $root.f.cmd2.dumpsql
		pack $root.f.cmd2.dumpsql -side left -padx 2

		# End of Widgets creation

		# Configure Tree
		$tree configure -selectmode browse
		$tree column create -width 80 -tag NAME -text Name
		$tree column create -width 70 -tag TYPE -text Type
		$tree column create -width 50 -tag PKEY -text PKey
		$tree column create  -width 50 -tag SORT -text Sort
		$tree column create  -width 30 -tag AUTOINCR -text AutoIncr
		$tree column create  -width 50 -tag UNIQUE -text Uniq
		$tree column create  -width 50 -tag NNULL -text NNull
		$tree column create  -width 70 -tag DEFVAL -text Default
		$tree column create  -width 80 -tag CHECK -text Check
		$tree column create  -width 50 -tag CHKCONFLICT -text CheckConflict
		$tree column create  -width 60 -tag COLLATE -text Collate
		# the next columns are invisible and keep the tree status.
		$tree column create -visible no -tag HASDEFAULT
		$tree column create -visible no -tag SEQ
		$tree column create -visible no -tag STATUS
		::Cmd::changeTreectrlHeaderLoolAndFeele $tree
		$tree state define new
		$tree state define modify
		$tree state define normal
		$tree state define pkey
		$tree state define autoincr
		$tree state define uniq
		$tree state define nnull
		$tree state define defisnull
		# this 3 states are used for setting text color.
		$tree state define pkeycolor
		$tree state define uniqcolor
		$tree state define nnullcolor
		$tree element create eStatusIcon image -image [list \
			$::img(columnNew)    new \
			$::img(columnEdit)   modify \
			$::img(columnNormal) normal ]
		$tree element create ePKIcon image -image [list $::img(checkPkey) pkey]
		$tree element create eAIIcon image -image [list $::img(checkAutoincr) autoincr]
		$tree element create eUQIcon image -image [list $::img(checkUnique) uniq]
		$tree element create eNNIcon image -image [list $::img(checkNNull) nnull]
		$tree element create eNullIcon image -image [list $::img(nullValue) defisnull]
		$tree element configure eTxt -fill [list \
			$::pref(tile_selectfg) selected \
			"#F95C6B" pkeycolor \
			"#41CC25" uniqcolor \
			"#6074C1" nnullcolor ]

		# create style
		set S [$tree style create STYLE_NAME]
		$tree style elements $S {eSel eStatusIcon eTxt}
		$tree style layout $S eStatusIcon -expand ns 
		$tree style layout $S eTxt -expand ns -squeeze x -padx {2 0}
		$tree style layout $S eSel -union {eStatusIcon eTxt} -iexpand news -detach yes
		foreach S {STYLE_PK STYLE_UQ STYLE_NN STYLE_DEFVAL} \
				I {ePKIcon eUQIcon eNNIcon eNullIcon} {\
			set S [$tree style create $S]
			$tree style elements $S [list eSel $I eTxt]
			$tree style layout $S $I   -expand ns -padx {1 0}
			$tree style layout $S eTxt -expand ns -padx {2 0} -squeeze x
			$tree style layout $S eSel -union [list $I eTxt] -iexpand news
		}
		set S [$tree style create STYLE_AINCR]
		$tree style elements $S {eSel eAIIcon}
		$tree style layout $S eAIIcon -expand news
		$tree style layout $S eSel -union {eAIIcon} -iexpand news -detach yes

		# bindings
		proc bind_click1 {x y} {
			variable tree
			set identify [$tree identify $x $y]
			if {[lindex $identify 0] eq ""} {
				if {[llength [$tree selection get]]} {
					$tree selection clear
				}
				return
			}
		}
		proc bind_dbclick1 {x y} {
			variable tree
			set identify [$tree identify $x $y]
			if {[lindex $identify 0] eq "item"} {
				modifyColumn
				return
			}
			if {[lindex $identify 0] eq ""} {
				addColumn
				return
			}
			if {[lindex $identify 0] eq "header"} {
				return
			}
		}
		bind $tree <Double-1> +[namespace code {bind_dbclick1 %x %y}]
		bind $tree <1>        +[namespace code {bind_click1 %x %y}]
		bind $tree <Return>    +[namespace code modifyColumn]
		bind $tree <BackSpace> +[namespace code deleteColumn]
		bind $tree <Delete>    +[namespace code deleteColumn]

		# root size
		eval [subst {wm minsize $root $info(minSize)}]
		::Util::updateWindowSize $root [join $info(iniSize) x]
		::Util::moveCenter $root $info(iniSize)
	}
	# Clean
	$tree item delete all
	$root.f.const.check delete 0 end
	$root.f.const.checkconflict set ""
	$root.f.const.checkconflict set "ON CONFLICT ..."
	$root.f.const.checkconflict configure -state "disabled"
	set dblist [Sqlite::getDBList writable]
	$root.f.table.db state !disabled
	$root.f.table.db configure -values $dblist

	switch -exact -- $info(runmode) {
		create {
			wm title $root [msgcat::mc "Create Table"]
			$root.f.table.db state !disabled
			set info(dbname)    [lindex $dblist 0]
			set info(tablename) ""
		}
		modify {
			wm title $root [msgcat::mc "Modify Table Schema"]
			$root.f.table.db state disabled
			setColumnFromSQLDict $tabledata
			if {[dict exists $tabledata constraint]} {
				set len [dict size [dict get $tabledata constraint]]
				for {set i 0} {$i < $len} {incr i} {
					set const [dict get $tabledata constraint $i]
					if {[dict get $const type] eq "check"} {
						set tableconst(check) [dict get $const value]
						if {[dict exists $const conflict]} {
							set tableconst(checkconflict) \
								[string toupper [dict get $const conflict]]
						}
						break
					}
				}
			}
			set info(oldtabledata) [dict create \
				dbname        $info(dbname) \
				tablename     $info(tablename) \
				column        [dumpTreeToColumnList $tree] \
				check         $tableconst(check) \
				checkconflict $tableconst(checkconflict) \
			]
		}
	}
};#>>>

proc GUICmd::TableBuilder::deleteColumn {} {;#<<<
	variable tree
	set item [$tree selection get]
	if {$item ne ""} {
		set next [$tree item nextsibling $item]
		set prev [$tree item prevsibling $item]
		$tree item delete $item
		if {$next ne ""} {
			set item $next
		} elseif {$prev ne ""} {
			set item $prev
		} else {
			return
		}
		$tree selection add $item
		$tree see $item
	}
};#>>>

proc GUICmd::TableBuilder::moveColumn {dir} {;#<<<
	variable tree
	set src [lindex [$tree selection get] 0]
	if {$src eq {}} return
	
	switch -exact -- $dir {
		up { set cmd prevsibling }
		down { set cmd nextsibling }
		default {return}
	}
	set tgt [$tree item $cmd $src]
	if {$tgt eq ""} {
		return
	}
	$tree item $cmd $tgt $src
	$tree see $src
};#>>>

# set Column from SQLParser's dict type table data.
proc GUICmd::TableBuilder::setColumnFromSQLDict {tabledata} {;#<<<
	# get table const
	set tconst [dict create \
		pkey [dict create] \
		unique [dict create]]
	if {[dict exists $tabledata constraint]} {
		set consts [dict get $tabledata constraint]
		set constslen [dict size $consts]
		for {set i 0} {$i < $constslen} {incr i} {
			set const [dict get $consts $i]
			set ctype [dict get $const type]
			switch -exact -- $ctype {
				pkey -
				unique {}
				default {continue}
			}
			if {[dict exists $const conflict]} {
				set conflict [dict get $const conflict]
			} else {
				set conflict 1
			}
			foreach {n col} [dict get $const column] {
				if {[dict exists $tconst $ctype $col]} {
					if {[dict get $tconst $ctype $col] == 1} {
						dict set tconst $ctype $col $conflict
					}
				} else {
					dict set tconst $ctype $col $conflict
				}
			}
		}
	}
	
	# Set the data
	set cols [dict get $tabledata column]
	set collen [dict size $cols]
	for {set i 0} {$i < $collen} {incr i} {
		set col [dict get $cols $i]
		set coldata [dict create]
		dict set coldata name [dict get $col name]
		dict set coldata type [expr {[dict exists $col type] ? [dict get $col type] : ""}]
		# constraint
		dict set coldata pkey 0
		dict set coldata sort ""
		dict set coldata autoincrement 0
		dict set coldata unique 0
		dict set coldata notnull 0
		dict set coldata hasdefault 0
		dict set coldata defaultvalue NULL
		dict set coldata check ""
		dict set coldata checkconflict ""
		dict set coldata collate ""
		dict set coldata seq $i
		dict set coldata status normal
		
		if {[dict exists $col constraint]} {
			set consts [dict get $col constraint]
			set constssize [dict size $consts]
			for {set ii 0} {$ii < $constssize} {incr ii} {
				set const [dict get $consts $ii]
				switch -exact -- [dict get $const type] {
					pkey {
						dict set coldata pkey 1
						if {[dict exists $const sort]} {
							dict set coldata sort [string toupper [dict get $const sort]]
						}
						if {[dict exists $const autoincrement]} {
							dict set coldata autoincrement 1
						}
						if {[dict exists $const conflict]} {
							dict set coldata pkey [string toupper [dict get $const conflict]]
						}
					}
					notnull {
						dict set coldata notnull 1
						if {[dict exists $const conflict]} {
							dict set coldata notnull [string toupper [dict get $const conflict]]
						}
					}
					unique {
						dict set coldata unique 1
						if {[dict exists $const conflict]} {
							dict set coldata unique [string toupper [dict get $const conflict]]
						}
					}
					check {
						dict set coldata check [dict get $const value]
						if {[dict exists $const conflict]} {
							dict set coldata checkconflict [string toupper [dict get $const conflict]]
						}
					}
					default {
						dict set coldata hasdefault 1
						dict set coldata defaultvalue [dict get $const value]
					}
					collate {
						dict set coldata collate [dict get $const value]
					}
					references {
					}
				}
			}
		}
		# Table constraints
		set colname [dict get $col name]
		if {[dict exists $tconst pkey $colname]} {
			if {![dict exists $coldata pkey]} {
				dict set coldata pkey 1
			}
			dict set coldata pkey [string toupper [dict get $tconst pkey $colname]]
		}
		if {[dict exists $tconst unique $colname]} {
			if {![dict exists $coldata unique]} {
				dict set coldata unique 1
			}
			dict set coldata unique [string toupper [dict get $tconst unique $colname]]
		}
		setColumn {} $coldata
	}
};#>>>

proc GUICmd::TableBuilder::setColumn {item data} {;#<<<
	variable info
	variable tree
	variable treeid
	variable conflicttype
	
	set conflicttext [dict create \
		"" "" 0 "" 1 "" ROLLBACK ROLLBACK \
		ABORT ABORT FAIL FAIL \
		IGNORE IGNORE REPLACE REPLACE]

	if { $item eq "" || $item <= 0 || [$tree item count] <= $item} {
		set isNewItem 1
	} else {
		set isNewItem 0
	}

	# create data tree
	if {[dict get $data pkey] eq "0"} {
		dict set data sort ""
		dict set data autoincrement 0
	} else {
		if {[lsearch $conflicttype [dict get $data pkey]] < 0} {
			dict set data pkey 1
		}
	}
	if {[dict get $data unique] ne "0"} {
		if {[lsearch $conflicttype [dict get $data unique]] < 0} {
			dict set data unique 1
		}
	}
	if {[dict get $data notnull] ne "0"} {
		if {[lsearch $conflicttype [dict get $data notnull]] < 0} {
			dict set data notnull 1
		}
	}
	
	#create item
	if {$isNewItem} {
		set item [$tree item create -open no]
		$tree item lastchild root $item
	} else {
		set item [$tree item id $item]
	}
	$tree item style set $item NAME     STYLE_NAME
	$tree item style set $item TYPE     STYLE_TEXT
	$tree item style set $item PKEY     STYLE_PK
	$tree item style set $item SORT     STYLE_TEXT
	$tree item style set $item AUTOINCR STYLE_AINCR
	$tree item style set $item UNIQUE   STYLE_UQ
	$tree item style set $item NNULL    STYLE_NN
	$tree item style set $item DEFVAL   STYLE_DEFVAL
	$tree item style set $item CHECK    STYLE_TEXT
	$tree item style set $item CHKCONFLICT STYLE_TEXT
	$tree item style set $item COLLATE     STYLE_TEXT
	$tree item style set $item HASDEFAULT  STYLE_TEXT
	$tree item style set $item SEQ         STYLE_TEXT
	$tree item style set $item STATUS      STYLE_TEXT
	#set text color
	$tree item state forcolumn $item PKEY     pkeycolor
	$tree item state forcolumn $item SORT     pkeycolor
	$tree item state forcolumn $item AUTOINCR pkeycolor
	$tree item state forcolumn $item UNIQUE uniqcolor
	$tree item state forcolumn $item NNULL  nnullcolor
	
	# set state
	set state ""
	switch -exact -- [dict get $data status] {
		new     {set state { new !modify !normal}}
		modify  {set state {!new  modify !normal}}
		normal  {set state {!new !modify  normal}}
		default {set state { new !modify !normal}}
	}
	lappend state [expr {[dict get $data pkey] ne "0" ? "pkey" : "!pkey"}]
	lappend state [expr {[dict get $data autoincrement] ? "autoincr" : "!autoincr"}]
	lappend state [expr {[dict get $data unique] ne "0" ? "uniq" : "!uniq"}]
	lappend state [expr {[dict get $data notnull] ne "0" ? "nnull" : "!nnull"}]
	lappend state [expr {[dict get $data hasdefault] ? "!defisnull" : "defisnull"}]
	$tree item state set $item $state

	# set text and data
	$tree item text $item NAME [dict get $data name]
	$tree item text $item TYPE [dict get $data type]
	$tree item element configure $item PKEY eTxt \
		-text [dict get $conflicttext [dict get $data pkey]] \
		-data [dict get $data pkey]
	$tree item text $item SORT [dict get $data sort]
	$tree item element configure $item UNIQUE eTxt \
		-text [dict get $conflicttext [dict get $data unique]] \
		-data [dict get $data unique]
	$tree item element configure $item NNULL eTxt \
		-text [dict get $conflicttext [dict get $data notnull]] \
		-data [dict get $data notnull]
	if {[dict get $data hasdefault]} {
		$tree item text $item DEFVAL [dict get $data defaultvalue]
	} else {
		$tree item text $item DEFVAL ""
	}
	$tree item text $item CHECK [dict get $data check]
	$tree item element configure $item CHKCONFLICT eTxt \
		-text [dict get $conflicttext [dict get $data checkconflict]] \
		-data [dict get $data checkconflict]
	$tree item text $item COLLATE [dict get $data collate]
	$tree item text $item HASDEFAULT [dict get $data hasdefault]
	$tree item text $item SEQ        [dict get $data seq]
	$tree item text $item STATUS     [dict get $data status]
};#>>>

# modify column data. 
# Called from item's Double-1 event or btn cmd.
proc GUICmd::TableBuilder::modifyColumn {} {;#<<<
	variable tree
	
	set item [lindex [$tree selection get] 0]
	if {$item eq {}} return
	
	# modify column
	set data [dumpTreeItemToDict $item]
	set result [setColumnDialog $data]
	$tree selection add $item

	# was data changed?
	if {[dict size $result] == 0} {return}
	if {[Util::dictCompare [dict remove $result status] [dict remove $data status] 0] == 0} {return}
	setColumn $item $result
};#>>>

proc GUICmd::TableBuilder::addColumn {} {;#<<<
	set data [setColumnDialog]
	if {[llength $data] > 0} {
		setColumn {} $data
	}
};#>>>

# GUI Dialog for column
proc GUICmd::TableBuilder::setColumnDialog { {data {}} } {;#<<<
	variable info
	variable root
	variable columnDialog
	variable columnDialogState
	variable conflicttype
	
	
	if {$columnDialog eq {}} {
		set columnDialog [toplevel .cmdTableBuilder_ColumnDialog]
		wm withdraw $columnDialog
		set cancelAction "wm withdraw $columnDialog
		set [namespace current]::columnDialogState cancel"
		set okAction "wm withdraw $columnDialog
		set [namespace current]::columnDialogState ok"
		wm protocol $columnDialog WM_DELETE_WINDOW $cancelAction
		wm title $columnDialog [msgcat::mc "Modify Column"]
		bind $columnDialog <Escape> $cancelAction

		ttk::frame $columnDialog.f -padding 4
		pack $columnDialog.f -expand yes -fill both 
		
		# Name DataType
		ttk::frame $columnDialog.f.n
		ttk::label $columnDialog.f.n.namel -text [msgcat::mc Name] -class Label
		ttk::entry $columnDialog.f.n.name
		ttk::label $columnDialog.f.n.typel -text [msgcat::mc "Data Type"] -class Label
		ttk::combobox $columnDialog.f.n.type \
		-values {INTEGER REAL FLOAT NUMERIC CHAR TEXT BOOLEAN DATETIME TIMESTAMP BLOB}

		grid $columnDialog.f.n.namel -row 0 -column 0 -sticky w
		grid $columnDialog.f.n.name  -row 0 -column 1 -sticky ew -pady 2
		grid $columnDialog.f.n.typel -row 1 -column 0 -sticky w
		grid $columnDialog.f.n.type  -row 1 -column 1 -sticky ew -pady 2
		grid columnconfigure $columnDialog.f.n 1 -weight 1
		pack $columnDialog.f.n -side top -fill x
		Util::bindEditorPopCommon $columnDialog.f.n.name
		Util::bindEditorPopCommon $columnDialog.f.n.type 
		bind $columnDialog.f.n.name <Return> +$okAction
		bind $columnDialog.f.n.type <Return> +$okAction
		
		# Column-Constraints
		set conflictlist [concat {{}} $conflicttype]
		set f [ttk::labelframe $columnDialog.f.c -text [msgcat::mc Constraints] -padding {4 0 4 4}]
		ttk::checkbutton $f.pkey -text "Primary Key"
		ttk::combobox    $f.pkeyconflict -state read -values $conflictlist
		ttk::label       $f.sortl    -text "Sort-Order"
		ttk::combobox    $f.sort -width 8 -state read -values {{} ASC DESC}
		ttk::checkbutton $f.autoincr -text "Auto Increment"

		ttk::checkbutton $f.unique -text "Unique"
		ttk::combobox    $f.uniqueconflict -state read -values $conflictlist
		
		ttk::checkbutton $f.notnull -text "Not Null"
		ttk::combobox    $f.notnullconflict -state read -values $conflictlist
		
		ttk::checkbutton $f.default -text "Default"
		ttk::entry       $f.defaultvalue
		Util::bindEditorPopCommon $f.defaultvalue

		# TODO: Refactor this "Check" widgets behavior
		ttk::label       $f.checkl -text "Check"
		ttk::entry       $f.check -textvariable ::$f.checkvalue
		ttk::combobox    $f.checkconflict -state read -values $conflictlist
		Util::bindEditorPopCommon $f.check

		ttk::label       $f.collatel -text "Collate"
		ttk::combobox    $f.collate -state write -values [::Sqlite::getCollationList $info(dbname)]
		Util::bindEditorPopCommon $f.collate

		grid $f.pkey            -row 0 -column 0 -sticky w
		grid $f.pkeyconflict    -row 0 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.sortl           -row 1 -column 0 -sticky w -padx {24 0}
		grid $f.sort            -row 1 -column 1 -sticky w -padx {4 0} -pady 2
		grid $f.autoincr        -row 2 -column 0 -sticky w -columnspan 2 -padx {24 0} -pady 2
		grid $f.unique          -row 3 -column 0 -sticky w
		grid $f.uniqueconflict  -row 3 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.notnull         -row 4 -column 0 -sticky w
		grid $f.notnullconflict -row 4 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.default         -row 5 -column 0 -sticky w
		grid $f.defaultvalue    -row 5 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.checkl          -row 6 -column 0 -sticky w
		grid $f.check           -row 6 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.checkconflict   -row 7 -column 1 -sticky ew -padx {4 0} -pady 2
		grid $f.collatel        -row 8 -column 0 -sticky w
		grid $f.collate         -row 8 -column 1 -sticky ew -padx {4 0} -pady 2

		grid columnconfigure $f 1 -weight 1
		pack $f -side top -fill x
		
		# Bindings
		bind $f.defaultvalue <Return> +$okAction
		bind $f.check        <Return> +$okAction
		bind $f.collate      <Return> +$okAction

		# $f is $columnDialog.f.c
		set $f.pkey 0
		trace add variable ::$f.pkey write [namespace current]::$f.pkey
		proc $f.pkey args [subst -nocommands {
			set s [expr {[set ::[lindex \$args 0]] ? "readonly":"disabled"}]
			$f.pkeyconflict configure -state \$s
			$f.autoincr     configure -state \$s
			$f.sortl        configure -state \$s
			$f.sort         configure -state \$s
		}]

		set ::$f.unique 0
		trace add variable ::$f.unique write [namespace current]::$f.unique
		proc $f.unique args [subst -nocommands {
			$f.uniqueconflict configure -state \
			[expr {[set ::[lindex \$args 0]] ? "readonly":"disabled"}]
		}]
		
		set ::$f.notnull 1
		trace add variable ::$f.notnull write [namespace current]::$f.notnull
		proc $f.notnull args [subst -nocommands {
			$f.notnullconflict configure -state \
			[expr {[set ::[lindex \$args 0]] ? "readonly":"disabled"}]
		}]

		set ::$f.default 1
		trace add variable ::$f.default write [namespace current]::$f.default
		proc $f.default args [subst -nocommands {
			$f.defaultvalue configure -state \
			[expr {[set ::[lindex \$args 0]] ? "normal":"disabled"}]
		}]

		set ::$f.checkvalue {}
		trace add variable ::$f.checkvalue write [namespace current]::$f.check
		proc $f.check args [subst -nocommands {
			set cmd [lindex \$args 0]
			if {\$cmd ne "::$f.checkvalue"} {return}
			set cvalue [string trim [set \$cmd]]
			if {[string length \$cvalue] > 0} {
				$f.checkconflict configure -state "readonly"
			} else {
				$f.checkconflict configure -state "disabled"
			}
		}]
		
		# OK Cancel
		ttk::frame $columnDialog.f.cmd
		ttk::button $columnDialog.f.cmd.ok     -text OK     -command $okAction
		ttk::button $columnDialog.f.cmd.cancel -text Cancel -command $cancelAction
		foreach w [winfo children $columnDialog.f.cmd] {
			Util::bindMouseEvent $w
			pack $w -padx 2 -side left -fill x -expand no
		}
		pack $columnDialog.f.cmd -side right -fill x -pady {4 0}

		wm resizable $columnDialog 1 0
		::Util::updateWindowSize $columnDialog
		::Util::moveCenter $columnDialog 
	}
	# Clear
	$columnDialog.f.n.name delete 0 end
	$columnDialog.f.n.type set {}
	set ::$columnDialog.f.c.pkey     0
	$columnDialog.f.c.pkeyconflict state disabled
	$columnDialog.f.c.autoincr     state disabled
	$columnDialog.f.c.sortl        state disabled
	$columnDialog.f.c.sort         state disabled
	$columnDialog.f.c.pkeyconflict set "ON CONFLICT ..."
	set ::$columnDialog.f.c.autoincr 0 
	$columnDialog.f.c.sort         set ""
	set ::$columnDialog.f.c.unique   0
	$columnDialog.f.c.uniqueconflict state disabled
	$columnDialog.f.c.uniqueconflict set "ON CONFLICT ..."
	set ::$columnDialog.f.c.notnull  0
	$columnDialog.f.c.notnullconflict state disabled
	$columnDialog.f.c.notnullconflict set "ON CONFLICT ..."
	set ::$columnDialog.f.c.default  1
	$columnDialog.f.c.defaultvalue state !disabled
	$columnDialog.f.c.defaultvalue delete 0 end
	set ::$columnDialog.f.c.checkvalue ""
	$columnDialog.f.c.checkconflict set "ON CONFLICT ..."
	$columnDialog.f.c.checkconflict state disabled
	$columnDialog.f.c.collate delete 0 end

	# Change title
	if {[dict size $data] > 0} {
		wm title $columnDialog [msgcat::mc "Modify Column"]
	} else {
		wm title $columnDialog [msgcat::mc "Add Column"]
	}

	# Set
	if {[dict size $data] > 0} {
		dputs {setColumnDialog  data=$data}
		$columnDialog.f.n.name insert 0 [dict get $data name]
		$columnDialog.f.n.type insert 0 [dict get $data type]

		if {[dict get $data pkey] eq "0"} {
			set ::$columnDialog.f.c.pkey 0
		} else {
			set ::$columnDialog.f.c.pkey 1
			set conflict [dict get $data pkey]
			if {[lsearch $conflicttype $conflict] >= 0} {
				$columnDialog.f.c.pkeyconflict set $conflict
			}
			set ::$columnDialog.f.c.autoincr [dict get $data autoincrement]
			$columnDialog.f.c.sort set [dict get $data sort]
		}

		if {[dict get $data unique] eq "0"} {
			set ::columnDialog.f.c.unique 0
		} else {
			set ::$columnDialog.f.c.unique 1
			set conflict [dict get $data unique]
			if {[lsearch $conflicttype $conflict] >= 0} {
				$columnDialog.f.c.uniqueconflict set $conflict
			}
		}
		if {[dict get $data notnull] eq "0"} {
			set ::$columnDialog.f.c.notnull 0
		} else {
			set ::$columnDialog.f.c.notnull 1
			set conflict [dict get $data notnull]
			if {[lsearch $conflicttype $conflict] >= 0} {
				$columnDialog.f.c.notnullconflict set $conflict
			}
		}

		if {[dict get $data hasdefault] == 1} {
			set ::$columnDialog.f.c.default 1
			$columnDialog.f.c.defaultvalue insert 0 [dict get $data defaultvalue]
		} else {
			set ::$columnDialog.f.c.default 0
		}
		
		set cvalue [string trim [dict get $data check]]
		if {[string length $cvalue] > 0} {
			set ::$columnDialog.f.c.checkvalue $cvalue
			set conflict [dict get $data checkconflict]
			if {[lsearch $conflicttype $conflict] >= 0} {
				$columnDialog.f.c.checkconflict set $conflict
			}
		}

		$columnDialog.f.c.collate insert 0 [dict get $data collate]
		
		$columnDialog.f.n.name selection range 0 end
	}
	
	# Show Dialog and Wait Event
	wm transient $columnDialog $root
	wm deiconify $columnDialog
	update
	::tk::SetFocusGrab $columnDialog ""

	focus -force $columnDialog.f.n.name

	tkwait var [namespace current]::columnDialogState
	::tk::RestoreFocusGrab $columnDialog "" withdraw

	if {$columnDialogState eq "cancel"} {
		return [dict create]
	}

	# Create Return Value
	set rdata [dict create \
		name [$columnDialog.f.n.name get] \
		type [$columnDialog.f.n.type get] \
	]
	if {[set ::$columnDialog.f.c.pkey] == 0} {
		dict set rdata pkey 0
		dict set rdata sort ""
		dict set rdata autoincrement 0
	} else {
		dict set rdata sort [$columnDialog.f.c.sort get]
		dict set rdata autoincrement [set ::$columnDialog.f.c.autoincr]
		set conflict [$columnDialog.f.c.pkeyconflict get]
		if {[lsearch $conflicttype $conflict] < 0} {
			dict set rdata pkey 1
		} else {
			dict set rdata pkey $conflict
		}
	}

	if {[set ::$columnDialog.f.c.unique] == 0} {
		dict set rdata unique 0
	} else {
		set conflict [$columnDialog.f.c.uniqueconflict get]
		if {[lsearch $conflicttype $conflict] < 0} {
			dict set rdata unique 1
		} else {
			dict set rdata unique $conflict
		}
	}

	if {[set ::$columnDialog.f.c.notnull] == 0} {
		dict set rdata notnull 0
	} else {
		set conflict [$columnDialog.f.c.notnullconflict get]
		if {[lsearch $conflicttype $conflict] < 0} {
			dict set rdata notnull 1
		} else {
			dict set rdata notnull $conflict
		}
	}

	if {[set ::$columnDialog.f.c.default] == 0} {
		dict set rdata hasdefault 0
		dict set rdata defaultvalue ""
	} else {
		dict set rdata hasdefault 1
		dict set rdata defaultvalue [$columnDialog.f.c.defaultvalue get]
	}

	set cvalue [string trim [set ::$columnDialog.f.c.checkvalue]]
	if {[string length $cvalue] > 0} {
		dict set rdata check $cvalue
		set conflict [$columnDialog.f.c.checkconflict get]
		if {[lsearch $conflicttype $conflict] < 0} {
			dict set rdata checkconflict ""
		} else {
			dict set rdata checkconflict $conflict
		}
	} else {
		dict set rdata check ""
		dict set rdata checkconflict ""
	}
	dict set rdata collate [$columnDialog.f.c.collate get]

	if {[dict exists $data seq]} {
		dict set rdata seq [dict get $data seq]
	} else {
		dict set rdata seq -1
	}

	if {[dict exists $data status]} {
		if {[dict get $data status] eq "normal"} {
			dict set rdata status "modify"
		} else {
			dict set rdata status [dict get $data status]
		}
	} else {
		dict set rdata status "new"
	}
	return $rdata
};#>>>

proc GUICmd::TableBuilder::runSQL {} {
	variable info
	# Create SQL Stetement
	if {$info(runmode) eq "create"} {
		set query [buildCreateQuery $info(dbname) $info(tablename)]
	} else {
		set query [buildModifyQuery $info(dbname) $info(tablename)]
	}
	# run SQL
	if {$info(showonly)} {
		Editor::setText $query
		return 0
	} else {
		if {[Sqlite::evalQuery $query] != 0} {
			Sqlite::evalQuery "ROLLBACK;"
			return 1
		}
		::Cmd::updateFrames
		::Table::clear
		return 0
	}
}

proc GUICmd::TableBuilder::buildCreateQuery {dbname tablename} {;#<<<
	variable tree
	variable conflicttype
	variable tableconst
	
	set sqliteversion [::Sqlite::getCurrentFullVersion]

	# Count Primary Key's columns. and if more than one column is
	# primary key, use table constraints.
	set pkeylist [list]
	set pkeylistconflict [list]
	set tabledata [dumpTreeToColumnList $tree]
	foreach {row} $tabledata {
		if {[dict get $row pkey] ne "0"} {
			lappend pkeylist '[dict get $row name]'
			lappend pkeylistconflict [string toupper [dict get $row pkey]]
		}
	}
	
	# Build SQL
	set coldeflist [list]
	foreach {row} $tabledata {
		foreach {n v} $row {set $n $v} ;# set all values

		set coldef '$name'

		append coldef " $type"

		if {[llength $pkeylist] == 1 && $pkey ne "0"} {
			append coldef " PRIMARY KEY"
			if {$sort ne ""} {
				append coldef " $sort"
			}
			if {[lsearch $conflicttype $pkey] >= 0} {
				append coldef " ON CONFLICT $pkey"
			}
			if {$autoincrement} {
				append coldef " AUTOINCREMENT"
			}
		}
		if {$unique ne "0"} {
			append coldef " UNIQUE"
			if {[lsearch $conflicttype $unique] >= 0} {
				append coldef " ON CONFLICT $unique"
			}
		}
		if {$notnull ne "0"} {
			append coldef " NOT NULL"
			if {[lsearch $conflicttype $notnull] >= 0} {
				append coldef " ON CONFLICT $notnull"
			}
		}
		if {$hasdefault == 1} {
			set defaultvalue [Util::getDefaultValue $sqliteversion $type $defaultvalue]
			append coldef " DEFAULT $defaultvalue"
		}
		set check [string trim $check]
		if {[string length $check] > 0} {
			append coldef " CHECK($check)"
			if {[lsearch $conflicttype $checkconflict] >= 0} {
				append coldef " ON CONFLICT $checkconflict"
			}
		}
		if {[string length $collate] > 0} {
			append coldef " COLLATE $collate"
		}
		lappend coldeflist $coldef
	};#end of foreach
	
	if {[Sqlite::getCurrentVersion] == 3} {
		set query "CREATE TABLE '$dbname'.'$tablename' (\n    "
	} else {
		if {$dbname eq "temp"} {
		set query "CREATE TEMP TABLE '$tablename' (\n    "
		} else {
		set query "CREATE TABLE '$tablename' (\n    "
		}
	}
	append query [join $coldeflist ",\n    "]

	if {[llength $pkeylist] > 1} {
		append query ",\n    PRIMARY KEY (" [join $pkeylist {, }] ")"
		if {[lsearch $conflicttype [lindex $pkeylistconflict 0]] >= 0} {
			append query " ON CONFLICT " [lindex $pkeylistconflict 0]
		}
	}
	if {[string trim $tableconst(check)] ne ""} {
		append query ",\n    CHECK($tableconst(check))"
		if {[lsearch $conflicttype $tableconst(checkconflict)] >= 0} {
			append query " ON CONFLICT $tableconst(checkconflict)"
		}
	}
	append query "\n);\n"
	return $query
};#>>>

proc GUICmd::TableBuilder::buildModifyQuery {dbname tablename} {;#<<<
	variable tree
	variable info
	variable tableconst
	variable conflicttype
	
	set sqliteversion [::Sqlite::getCurrentFullVersion]

	# Basically We can't use ALTER. So we must create sql such like next.
	#
	# BEGIN TRANSACTION;
	# CREATE TEMPORARY TABLE thoge(c1 , c2);
	# INSERT INTO thoge SELECT c1, c2 FROM hoge;
	# DROP TABLE hoge;
	# CREATE TABLE hoge(c1, c2);
	# INSERT INTO hoge SELECT c1, c2 FROM thoge;
	# DROP TABLE thoge;
	# COMMIT;
	
	set tabledata [dumpTreeToColumnList $tree]
	set changed(column) [didTableChanged [dict get $info(oldtabledata) column] $tabledata]
	set changed(check) [expr {[string trim [dict get $info(oldtabledata) check]] ne [string trim $tableconst(check)]}]
	set changed(tablename) [expr {[dict get $info(oldtabledata) tablename] ne $tablename}]

	# if there isnot change, return empty
	if {!($changed(column) || $changed(check) || $changed(tablename))} {
		return ""
	}

	# if only table name was changed and other wasn't changed, 
	# we can use ALTER TABLE on sqlite3.(not sqlite2)
	if { !$changed(column) && !$changed(check) && $changed(tablename) &&
		[Sqlite::getCurrentVersion] == 3} {
		set oldconflict [dict get $info(oldtabledata) checkconflict]
		set newconflict $tableconst(checkconflict)
		if {[lsearch $conflicttype $oldconflict] < 0} { set oldconflict "" }
		if {[lsearch $conflicttype $newconflict] < 0} { set newconflict "" }
		if {$oldconflict eq $newconflict} {
			set db    [dict get $info(oldtabledata) dbname]
			set table [dict get $info(oldtabledata) tablename]
			return "ALTER TABLE '$db'.'$table' RENAME TO '$tablename';"
		}
	}
	
	if {[Sqlite::getCurrentVersion] == 3} {
		set oldname "'[dict get $info(oldtabledata) dbname]'.'[dict get $info(oldtabledata) tablename]'"
		set newname "'$dbname'.'$tablename'"
	} else {
		set oldname "'[dict get $info(oldtabledata) tablename]'"
		set newname "'$tablename'"
	}
	set tempname "'tksqlite_temp_${tablename}_[clock seconds]'"
	
	# create temp table sql
	set tempcreatesql "CREATE TEMP TABLE $tempname (\n"
	set changelist [list]
	set tempcols [list]
	foreach {row} $tabledata {
		lappend tempcols '[dict get $row name]'
	}
	append tempcreatesql {    } [join $tempcols ", "] \n
	append tempcreatesql ");\n"
	
	# create temp table insert sql
	set tempinsertsql "INSERT INTO 'temp'.$tempname\nSELECT "
	set oldcols [list]
	foreach {row} [dict get $info(oldtabledata) column] {
		lappend oldcols \"[dict get $row name]\"
	}
	set selectcols [list]
	foreach {row} $tabledata {
		set seq [dict get $row seq]
		if {$seq < 0} {
			if {[dict get $row hasdefault] == 1} {
				set type [dict  get $row type]
				set defaultvalue [dict get $row defaultvalue]
				set defaultvalue [Util::getDefaultValue $sqliteversion $type $defaultvalue]
				lappend selectcols $defaultvalue
				continue
			}
			if {[dict get $row notnull] == 1} {
				lappend selectcols "''"
			} else {
				lappend selectcols NULL
			}
		} else {
			lappend selectcols [lindex $oldcols $seq]
		}
	}
	append tempinsertsql "[join $selectcols {, }] FROM $oldname;\n";
	
	# build all query
	set query ""
	append query "BEGIN TRANSACTION;\n\n"
	append query $tempcreatesql\n
	append query $tempinsertsql\n
	append query "DROP TABLE $oldname;\n"
	append query [buildCreateQuery $dbname $tablename]\n
	append query "INSERT INTO $newname \nSELECT * FROM $tempname;\n\n"
	append query "DROP TABLE $tempname;\n\n"
	append query "COMMIT;\n"
	return $query
};#>>>

proc GUICmd::TableBuilder::dumpTreeItemToDict {item} {;#<<<
	variable tree

	set data [dict create]
	dict set data name [$tree item text $item NAME]
	dict set data type [$tree item text $item TYPE]
	dict set data pkey [$tree item element cget $item PKEY eTxt -data]
	dict set data sort [$tree item text $item SORT]
	dict set data autoincrement [$tree item state get $item autoincr]
	dict set data unique        [$tree item element cget $item UNIQUE eTxt -data]
	dict set data notnull       [$tree item element cget $item NNULL eTxt -data]
	dict set data defaultvalue  [$tree item text $item DEFVAL]
	dict set data check         [$tree item text $item CHECK]
	dict set data checkconflict [$tree item element cget $item CHKCONFLICT eTxt -data]
	dict set data collate       [$tree item text $item COLLATE]
	dict set data hasdefault [$tree item text $item HASDEFAULT]
	dict set data seq        [$tree item text $item SEQ]
	dict set data status     [$tree item text $item STATUS]
	return $data
};#>>>

proc GUICmd::TableBuilder::dumpTreeToColumnList {tree} {;#<<<
	set clist ""
	set item [$tree item firstchild root]
	while {$item ne ""} {
		lappend clist [dumpTreeItemToDict $item]
		set item [$tree item nextsibling $item]
	}
	return $clist
};#>>>

proc GUICmd::TableBuilder::didTableChanged {old new} {
	if {[llength $old] ne [llength $new]} {
		return 1
	}
	foreach col1 $old col2 $new {
		if {[Util::dictCompare $col1 $col2 0]} {
			return 1
		}
	}
	return 0
}
;#>>>
# Create Index
namespace eval GUICmd::CreateIndex {;#<<<
	variable state {}

	variable info
	array set info {
		version {}
		dbname {}
		tablename {}
		indexname {}
		showonly 0
		uniq 0
	}
	# Widgets
	variable root {}
	variable tree
	variable dcombo
	variable tcombo
	variable iniSize {250 300}
}

proc GUICmd::CreateIndex::run {} {
	variable iniSize
	variable info
	variable state
	variable root
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	set info(version) [Sqlite::getCurrentVersion]
	_init
	# Init 
	setCurrentTable

	wm transient $root .
	::tk::SetFocusGrab $root $root
	wm deiconify $root
	focus -force $root.f.index
	raise $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state ne "cancel"} {Cmd::updateFrames}
}

proc GUICmd::CreateIndex::setCurrentTable {} {
	variable dcombo
	variable tcombo
	set itemData [lindex [::Dbtree::getSelection] 0]
	if {[llength $itemData] == 0} return 
	foreach {text image data} $itemData {break}
	if {[llength $data] < 4} {return}
	set db    [lindex $data 0]
	set table [lindex $data 2]
	if {$db ne {}} {
		set n [lsearch -exact [$dcombo cget -values] $db]
		if {$n >= 0} {
			$dcombo current $n
			initTableList
			set n [lsearch -exact [$tcombo cget -values] $table]
			if {$n >= 0} {
				$tcombo current $n
				initColumnList
			}
		}
	}
}

proc GUICmd::CreateIndex::_init {} {
	# widgets pointer
	variable tree
	variable root
	variable dcombo
	variable tcombo
	# vars
	variable info
	variable iniSize

	if {$root eq {}} {
		set root [toplevel .cmdCreateIndex]
		wm withdraw $root
		set cancelAction [namespace code {
				wm withdraw $root; set state cancel}]
		wm title $root [msgcat::mc "Create Index"]
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		bind $root <Escape> $cancelAction
		ttk::frame $root.f -padding 4
		pack $root.f -expand yes -fill both
		
		ttk::label $root.f.dbl -text [msgcat::mc Database]
		ttk::combobox $root.f.db -width 16 -state readonly \
			-textvariable [namespace current]::info(dbname)
		ttk::label $root.f.tablel -text [msgcat::mc Table]
		ttk::combobox $root.f.table -width 16 -state readonly \
			-textvariable [namespace current]::info(tablename)
		ttk::label $root.f.indexl -text [msgcat::mc Index] -class Label
		ttk::entry $root.f.index -textvariable  [namespace current]::info(indexname)
		ttk::checkbutton $root.f.uniq -text Unique -variable [namespace current]::info(uniq)
		Util::bindMouseEvent $root.f.uniq
		Util::bindEditorPopCommon $root.f.index	
		ttk::frame $root.f.clist
		ttk::frame $root.f.cmd
		set dcombo $root.f.db 
		set tcombo $root.f.table

		grid $root.f.dbl    -row 0 -column 0 -sticky w -padx {0 8}
		grid $root.f.db     -row 0 -column 1 -sticky ew -pady 2
		grid $root.f.tablel -row 1 -column 0 -sticky w
		grid $root.f.table  -row 1 -column 1 -sticky ew -pady 2
		grid $root.f.indexl -row 2 -column 0 -sticky w
		grid $root.f.index  -row 2 -column 1 -sticky ew -pady 2
		grid $root.f.uniq   -row 3 -column 0 -columnspan 2 -sticky ew -pady 2
		grid $root.f.clist  -row 4 -column 0 -columnspan 2 -sticky news -pady 2
		grid $root.f.cmd    -row 5 -column 0 -columnspan 2 -sticky ew -pady 2
		grid columnconfigure $root.f 1 -weight 1
		grid rowconfigure    $root.f 4 -weight 1
		
		Cmd::createTreectrl $root.f.clist y
		set tree $root.f.clist.tlist

		pack $root.f.clist.y -side right -fill y
		pack $root.f.clist.tlist -side left -fill both -expand yes
		
		ttk::checkbutton $root.f.cmd.showonly -text [msgcat::mc "Show Only"] \
			-variable [namespace current]::info(showonly)
		ttk::button $root.f.cmd.ok     -text OK
		ttk::button $root.f.cmd.cancel -text Cancel
		foreach w [winfo children $root.f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $root.f.cmd.showonly -side top   -pady 2 -fill x 
		pack $root.f.cmd.cancel   -side right -padx 2
		pack $root.f.cmd.ok       -side right -padx 2
		
		# Tree Configure
		$tree configure -selectmode browse
		$tree column create -tag SELECT -text Select -width 20
		$tree column create -tag NAME -text Name -width 80
		$tree column create -tag TYPE -text Type -width 40
		$tree column create -tag PK -text PK -width 30
		$tree column create -tag UQ -text UQ -width 30
		$tree column create -tag NN -text NN -width 30
		::Cmd::changeTreectrlHeaderLoolAndFeele $tree
		$tree notify bind $tree <Selection> \
		"::Cmd::changeSelectionStyleInTreectrl $tree \
		SELECT eWin -window CheckInTreeSelected CheckInTree"
		$tree state define haspk
		$tree state define hasuq
		$tree state define hasnn
		$tree element create ePKIcon image -image [list $::img(checkPkey) haspk]
		$tree element create eUQIcon image -image [list $::img(checkUnique) hasuq]
		$tree element create eNNIcon image -image [list $::img(checkNNull) hasnn]
		foreach S {STYLE_IDXPK STYLE_IDXUQ STYLE_IDXNN} \
				I {ePKIcon eUQIcon eNNIcon} {
			set S [$tree style create $S]
			$tree style elements $S [list eSel $I]
			$tree style layout $S $I -expand news
			$tree style layout $S eSel -union $I -iexpand news
		}
		
		# root size
		wm geometry $root [join $iniSize x]
		eval wm minsize $root $iniSize
		
		# bindings
		$dcombo configure -postcommand [namespace code {set info(pre_dbname) $info(dbname)}]
		bind $dcombo <<ComboboxSelected>> [namespace code {
			if {$info(pre_dbname) ne $info(dbname)} {
				initTableList
			}
		}]
		$tcombo configure -postcommand [namespace code {set info(pre_tablename) $info(tablename)}]
		bind $tcombo <<ComboboxSelected>> [namespace code {
			if {$info(pre_tablename) ne $info(tablename)} {
				initColumnList
			}
		}]
		$root.f.cmd.ok configure -command [namespace code {
			if {[runSQL]==0} {wm withdraw $root; set state ok}}]
		$root.f.cmd.cancel configure -command $cancelAction
		::Util::moveCenter $root $iniSize
	}
	# Clear
	initDBList
	set info(indexname) ""
	set info(uniq) 0
}

proc GUICmd::CreateIndex::initDBList {} {
	variable dcombo
	$dcombo configure -values [Sqlite::getDBList writable]
	$dcombo current 0
	initTableList
}

proc GUICmd::CreateIndex::initTableList {} {
	variable tcombo
	variable info

	set tlist [list]
	if {$info(dbname) eq ""} {
		set tlist [list]
	} else {
		set tlist [Sqlite::getTableList $info(dbname)]
	}
	$tcombo configure -values [lsort -dictionary $tlist]
	if {[llength $tlist] > 0} {
		$tcombo current 0
	}
	initColumnList
}

proc GUICmd::CreateIndex::initColumnList {} {
	variable tree
	variable info

	# clear tree
	$tree item delete all

	# get Column Data
	foreach d [Sqlite::getColumnList $info(dbname) $info(tablename)] {
		set check [ttk::checkbutton $tree.check[clock clicks] -style CheckInTree]
		bind $check <Destroy> {catch {unset %W}}
		uplevel #0 set $check 0

		dict with d {};#set values
		set item [$tree item create -open no]
		$tree item style set $item SELECT STYLE_WIN
		$tree item style set $item NAME STYLE_TEXT
		$tree item style set $item TYPE STYLE_TEXT
		$tree item style set $item PK STYLE_IDXPK
		$tree item style set $item UQ STYLE_IDXUQ
		$tree item style set $item NN STYLE_IDXNN
		set state ""
		if {$pkey}   { lappend state haspk }
		if {$unique} { lappend state hasuq }
		if {$null}   { lappend state hasnn }
		$tree item element configure $item SELECT eWin -window $check
		$tree item state set $item $state
		$tree item text $item NAME $name
		$tree item text $item TYPE $type
		$tree item lastchild root $item
	}
}

proc GUICmd::CreateIndex::runSQL {} {
	variable tree
	variable info

	set query "CREATE"
	if {$info(uniq)} {append query " UNIQUE"}
	
	switch -exact -- $info(version) {
		2 {
			append query " INDEX '$info(indexname)'"
			append query "\nON '$info(tablename)' ("
		}
		3 {
			append query " INDEX '$info(dbname)'.'$info(indexname)'"
			append query "\nON '$info(tablename)' ("
		}
		default {
			append query " INDEX '$info(dbname)'.'$info(indexname)'"
			append query "\nON '$info(tablename)' ("
		}
	}

	set columns [list]
	foreach item [$tree item children root] {
		set win [$tree item element cget $item SELECT eWin -window]
		if {![uplevel #0 set $win]} {continue}
		lappend columns "'[$tree item text $item NAME]'"
	}
	append query "[join $columns {, }]);"

	if {$info(showonly)} {
		Editor::setText $query
		return 0
	}
	return [Sqlite::evalQuery $query]
}
;#>>>
# Create Trigger
namespace eval GUICmd::CreateTrigger {;#<<<
	variable state ""

	variable root ""
	variable dcombo ""
	variable tcombo ""
	variable etimecombo ""
	variable coltree ""
	variable actions ""

	variable info
	array set info {
		dbname ""
		tablename ""
		showonly 0
		version  ""
		triggername ""
		eventtime ""
		eventtype ""
		tablelist ""
		viewlist  ""
	}
}

proc GUICmd::CreateTrigger::run {} {
	variable state
	variable root
	variable triggername
	variable dcombo
	variable tcombo
	variable etimecombo
	variable coltree
	variable info

	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	set info(version) [Sqlite::getCurrentVersion]

	_init
	setCurrentTable

	wm transient $root .
	::tk::SetFocusGrab $root $root
	wm deiconify $root
	focus -force $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state ne "cancel"} {Cmd::updateFrames}
}

proc GUICmd::CreateTrigger::_init {} {
	variable root
	variable triggername
	variable dcombo
	variable tcombo
	variable etimecombo
	variable coltree
	variable actions
	variable info

	if {$root eq ""} {
		set root [toplevel .cmdCreateTrigger]
		wm withdraw $root
		set cancelAction [namespace code {
				wm withdraw $root; set state cancel}]
		wm title $root [msgcat::mc "Create Trigger"]
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		bind $root <Escape> $cancelAction
		ttk::frame $root.f -padding 4
		pack $root.f -expand yes -fill both
		
		# Options
		ttk::label $root.f.dbl -text [msgcat::mc Database]
		ttk::combobox $root.f.db -width 16 -state readonly \
			-textvariable [namespace current]::info(dbname)
		ttk::label $root.f.tablel -text [msgcat::mc Table]/[msgcat::mc View]
		ttk::combobox $root.f.table -width 16 -state readonly \
			-textvariable [namespace current]::info(tablename)
		ttk::label $root.f.triggerl -text [msgcat::mc Trigger]
		ttk::entry $root.f.trigger -width 16 \
			-textvariable [namespace current]::info(triggername)
		ttk::label $root.f.eventl -text [msgcat::mc Event]
		ttk::combobox $root.f.eventtime -width 16 -state readonly \
			-textvariable [namespace current]::info(eventtime)
		ttk::combobox $root.f.eventtype -width 16 -state readonly \
			-textvariable [namespace current]::info(eventtype)
		ttk::label $root.f.collistl -text [msgcat::mc "Columns for 'UPDATE OF'"]
		Util::bindEditorPopCommon $root.f.trigger

		# Columns for "UPDATE OF"
		::Cmd::createTreectrl $root.f.collist y
		set tree $root.f.collist.tlist
		$tree configure -height 20
		$tree column create -width 50  -tag SELECT -text Select
		$tree column create -width 100 -tag NAME   -text Name
		$tree column create -width 100 -tag TYPE   -text Type
		::Cmd::changeTreectrlHeaderLoolAndFeele $tree
		$tree notify bind $tree <Selection> \
		"::Cmd::changeSelectionStyleInTreectrl $tree \
		SELECT eWin -window CheckInTreeSelected CheckInTree"

		# Action Text Enty
		ttk::frame $root.f.action -style TEntry -border 2
		ttk::scrollbar $root.f.action.y -command "$root.f.action.text yview"
		::SQLText::new $root.f.action.text
		$root.f.action.text configure -width 50 -height 10\
			-yscrollcommand "$root.f.action.y set"
		grid $root.f.action.text -row 0 -column 0 -sticky news
		grid $root.f.action.y    -row 0 -column 1 -sticky ns
		grid columnconfigure $root.f.action 0 -weight 1
		grid rowconfigure    $root.f.action 0 -weight 1
		Util::bindEditorPopCommon $root.f.action.text
		Cmd::bindTextFocusTtkEntry $root.f.action.text $root.f.action

		# Ok Cancel Buttons
		ttk::frame $root.f.cmd
		ttk::checkbutton $root.f.cmd.showonly -text [msgcat::mc "Show Only"] \
			-variable [namespace current]::info(showonly)
		ttk::button $root.f.cmd.ok      -text OK
		ttk::button $root.f.cmd.cancel  -text Cancel
		ttk::button $root.f.cmd.dumpsql -text [msgcat::mc "Confirm Query"]
		foreach w [winfo children $root.f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $root.f.cmd.cancel   -side right -padx 2
		pack $root.f.cmd.ok       -side right -padx 2
		pack $root.f.cmd.showonly -side right  -pady 2 -fill x 
		pack $root.f.cmd.dumpsql  -side left -padx 2
		
		# Place widgets
		grid $root.f.dbl      $root.f.db        $root.f.collistl
		grid $root.f.tablel   $root.f.table     $root.f.collist
		grid $root.f.triggerl $root.f.trigger   ^
		grid $root.f.eventl   $root.f.eventtime ^
		grid x                $root.f.eventtype ^
		grid $root.f.action   -                 -
		grid $root.f.cmd      -                 -
		grid configure $root.f.dbl $root.f.tablel $root.f.triggerl \
		$root.f.eventl -sticky w
		grid configure $root.f.db $root.f.table $root.f.trigger \
		$root.f.eventtime $root.f.eventtype -sticky ew  -pady 2 -padx 2
		grid configure $root.f.collistl -sticky w -padx {8 0}
		grid configure $root.f.collist -sticky news -padx {8 0}
		grid configure $root.f.action $root.f.cmd -sticky news -pady {8 0}
		grid columnconfigure $root.f 0 -weight 0
		grid columnconfigure $root.f 1 -weight 1
		grid columnconfigure $root.f 2 -weight 2
		grid columnconfigure $root.f {1 2} -uniform $root.f.group1
		grid rowconfigure $root.f 5 -weight 1
		pack $root.f
		
		# Bindings and Event Procs
		set triggername $root.f.trigger
		set dcombo $root.f.db    
		set tcombo $root.f.table 
		set etimecombo $root.f.eventtime
		set coltree $root.f.collist.tlist
		set actions $root.f.action.text
		$root.f.db configure -postcommand [namespace code {set info(pre_dbname) $info(dbname)}]
		bind $root.f.db <<ComboboxSelected>> [namespace code {
			if {$info(dbname) ne $info(pre_dbname)} {
				initTableList
			}
		}]
		$root.f.table configure -postcommand [namespace code {set info(pre_tablename) $info(tablename)}]
		bind $root.f.table <<ComboboxSelected>> [namespace code {
			if {$info(tablename) ne $info(pre_tablename)} {
				initEventTime; initColumnList
			}
		}]
		$root.f.eventtype configure -postcommand [namespace code {set info(pre_eventtype) $info(eventtype)}]
		bind $root.f.eventtype <<ComboboxSelected>> [namespace code {
			if {$info(eventtype) ne $info(pre_eventtype)} {
				initColumnList
			}
		}]
		$root.f.eventtype configure -values {"DELETE" "INSERT" "UPDATE" "UPDATE OF"}
		$root.f.cmd.ok configure -command [namespace code {
			if {[runSQL] == 0} {wm withdraw $root; set state ok}
		}]
		$root.f.cmd.dumpsql configure -command [namespace code {
			::Message::show -type appinfo \
			-message "SQL Query" \
			-detail [buildCreateQuery] \
			-parent $root
		}]
		$root.f.cmd.cancel configure -command $cancelAction

		set info(eventtime) ""
		set info(eventtype) "UPDATE"
		::tk::PlaceWindow $root widget .
	}
	initDBList
	set info(triggername) ""
	$actions delete 1.0 end
	$actions insert 1.0 "BEGIN\n\nEND"
	$actions mark set insert 2.0
	::SQLText::highlight $actions 1.0 end
	focus $triggername

}

proc GUICmd::CreateTrigger::setCurrentTable {} {
	variable dcombo
	variable tcombo
	set itemData [lindex [::Dbtree::getSelection] 0]
	if {[llength $itemData] == 0} return 
	foreach {text image data} $itemData {break}
	if {[llength $data] < 4} {return}
	set db    [lindex $data 0]
	set table [lindex $data 2]
	if {$db ne {}} {
		set n [lsearch -exact [$dcombo cget -values] $db]
		if {$n >= 0} {
			$dcombo current $n
			initTableList
			set n [lsearch -exact [$tcombo cget -values] $table]
			if {$n >= 0} {
				$tcombo current $n
				initEventTime
				initColumnList
			}
		}
	}
}

# returns table or view or unknown
proc GUICmd::CreateTrigger::getTableType {name} {
	variable info

	set name [string toupper $name]
	if {[lsearch [string toupper $info(tablelist)] $name] > -1} {
		return table
	}
	if {[lsearch [string toupper $info(viewlist)] $name] > -1} {
		return view
	}
	return unknown
}

proc GUICmd::CreateTrigger::initDBList {} {
	variable dcombo
	$dcombo configure -values [Sqlite::getDBList writable]
	$dcombo current 0
	initTableList
}

proc GUICmd::CreateTrigger::initTableList {} {
	variable tcombo
	variable info
	set dbname $info(dbname)
	set tlist [list]
	if {$dbname eq ""} {
		set tlist [list]
	} else {
		set info(tablelist) [lsort -dictionary [Sqlite::getTableList $dbname]]
		set info(viewlist)  [lsort -dictionary [Sqlite::getViewList $dbname]]
		set tlist [concat $info(tablelist) $info(viewlist)]
	}
	set info(tablename) ""
	$tcombo configure -values $tlist
	if {[llength $tlist] > 0} {
		$tcombo current 0
	}
	initEventTime
	initColumnList
}

proc GUICmd::CreateTrigger::initEventTime {} {
	variable info
	variable etimecombo
	set tabletype [getTableType $info(tablename)]

	if {$tabletype eq "unknown"} {
		$etimecombo configure -values ""
		set info(eventtime) ""
		return
	}
	if {$tabletype eq "view"} {
		$etimecombo configure -values [list "INSTEAD OF"]
		$etimecombo current 0
		return
	}
	if {$tabletype eq "table"} {
		$etimecombo configure -values {"" BEFORE AFTER}
		if {$info(eventtime) eq "INSTEAD OF"} {
			$etimecombo current 0
		}
		return
	}
}

proc GUICmd::CreateTrigger::initColumnList {} {
	variable info
	variable coltree
	set tree $coltree
	
	$tree item delete all
	if {$info(eventtype) ne "UPDATE OF"} {return}
	if {$info(dbname) eq "" || $info(tablename) eq ""} {return}

	foreach d [Sqlite::getColumnList $info(dbname) $info(tablename)] {
		set check $tree.check[clock clicks]
		set check [ttk::checkbutton $check -style CheckInTree]
		uplevel #0 set $check 0
		bind $check <Destroy> {catch {unset %W}}

		set item [$tree item create]
		$tree item style set $item SELECT STYLE_WIN
		$tree item style set $item NAME   STYLE_TEXT
		$tree item style set $item TYPE   STYLE_TEXT
		$tree item element configure $item SELECT eWin -window $check
		$tree item text $item NAME [dict get $d name]
		$tree item text $item TYPE [dict get $d type]
		$tree item lastchild root $item
	}
}

proc GUICmd::CreateTrigger::runSQL {} {
	variable info
	set query [buildCreateQuery]

	if {$info(showonly)} {
		Editor::setText $query
		return 0
	} else {
		if {[Sqlite::evalQuery $query] != 0} {
			return 1
		}
		::Cmd::updateFrames
		return 0
	}
}

proc GUICmd::CreateTrigger::buildCreateQuery {} {
	variable root
	variable coltree
	variable actions
	variable info

	set query "CREATE"

	if {$info(dbname) eq "temp"} {
		append query " TEMP TRIGGER"
	} else {
		append query " TRIGGER"
	}

	append query " '$info(triggername)'\n"
	
	append query "$info(eventtime)"
	if {$info(eventtime) ne ""} {
	append query "\n"
	}
	
	if {$info(eventtype) eq "UPDATE OF"} {
		append query "$info(eventtype)"
		set tree $coltree
		set cols ""
		foreach item [$tree item children root] {
			set win [$tree item element cget $item SELECT eWin -window]
			if {![uplevel #0 set $win]} {continue}
			lappend cols "'[$tree item text $item NAME]'"
		}
		append query " [join $cols {, }] ON"
	} else {
		append query "$info(eventtype) ON"
	}

	switch -exact -- $info(version) {
		2 { append query " '$info(tablename)'\n" }
		3 -
		default { append query " '$info(dbname)'.'$info(tablename)'\n" }
	}
	append query [$actions get 1.0 {end -1 c}]
	return $query
}

;#>>>
# Search 
proc GUICmd::Search {} {;#<<<
	return [Tool::Searcher::run]
};#>>>
# Preference
namespace eval GUICmd::Preference {;#<<<
	# widget name
	variable root .cmdPreference
	variable enctree {}
	variable sqlite_ext_tree {}

	# vars
	variable iniSize {340 380}
	variable state
	variable info
	array set info {
		sqlite_ext_dir ""
	}
}

proc GUICmd::Preference::run {} {
	variable iniSize
	variable state
	variable root
	variable info

	setVal
	_init

	# Init 
	wm transient $root .
	::tk::SetFocusGrab $root $root
	if {[winfo width $root] == 1} {
		::Util::moveCenter $root $iniSize
	} else {
		::Util::moveCenter $root
	}
	wm deiconify $root
	focus -force $root
	raise $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state eq "ok"} {
		getVal
		Cmd::updateFrames
	}
}


proc GUICmd::Preference::_init {} {
	variable info
	variable iniSize
	variable root
	variable enctree
	variable sqlite_ext_tree

	if {![winfo exists $root]} {
		set root [toplevel $root]
		wm title $root [msgcat::mc Preference]
		set f [ttk::frame $root.f -padding 4]
		pack $f -expand yes -fill both
		
		# OK / Cancel Button
		ttk::frame $f.cmd
		ttk::button $f.cmd.ok     -text OK
		ttk::button $f.cmd.cancel -text Cancel
		foreach w [winfo children $f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $f.cmd -fill x -side bottom -pady 4
		pack $f.cmd.cancel $f.cmd.ok -side right -padx 2
		
		# Root binding
		set cancelAction [namespace code {wm withdraw $root;set state cancel}]
		set okAction [namespace code {wm withdraw $root;set state ok}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$f.cmd.ok     configure -command $okAction
        $f.cmd.cancel configure -command $cancelAction
		
		# Notebook
		set note [ttk::notebook $f.note]
		pack $note -side top -fill both -expand yes

		# Notebook/General
		#----------------------------------------------------------------
		set f [ttk::frame $note.general -padding 4]
		$note add $f -text [msgcat::mc General] -sticky news

		# (Table)
		ttk::labelframe $f.table -padding 2 -text [msgcat::mc "Grid Table"]
		ttk::checkbutton $f.table.vi -text [msgcat::mc "Use Vi like cursor move"] \
			-variable [namespace current]::info(table,vi)
		pack $f.table.vi -side left
		pack $f.table -fill x
		# (History)
		ttk::labelframe $f.history -padding 4 -text [msgcat::mc "Command History"]
		ttk::label $f.history.maxl -text "[msgcat::mc {Max History}] : "
		ttk::entry $f.history.max -width 5 \
			-textvariable [namespace current]::info(history,max) \
			-validate key -validatecommand {string is digit %S}
		pack $f.history.maxl $f.history.max -side left
		pack $f.history -fill x
		bind $f.history.max <Up>   {set [%W cget -textvariable] [expr {[%W get] + 10}]}
		bind $f.history.max <Down> {set [%W cget -textvariable] [expr {([%W get] - 10)<=0?0:([%W get] - 10)}]}
		
		# Notebook/Editor
		#----------------------------------------------------------------
		set f [ttk::frame $note.editor -padding 4]
		$note add $f -text [msgcat::mc "SQL Editor"] -sticky news

		ttk::labelframe $f.font -padding 4 -text [msgcat::mc Font]
		ttk::label $f.font.namel -text [msgcat::mc "Font Name"]
		ttk::combobox $f.font.name -state readonly \
			-values [lsort -dictionary [font families]] \
			-textvariable [namespace current]::info(sqltext,fontname)
		ttk::label $f.font.sizel -text [msgcat::mc Size]
		ttk::combobox $f.font.size -width 3 -state read -values {8 10 12 14 16 18 20 24} \
			-textvariable [namespace current]::info(sqltext,fontsize)
		pack $f.font.size  -side right -padx 2 -fill x
		pack $f.font.sizel -side right -padx 2
		pack $f.font.namel -side left -padx 2
		pack $f.font.name  -side left -padx 2 -fill x -expand yes
		pack $f.font -fill x

		ttk::labelframe $f.style -padding 4 -text [msgcat::mc "String Style"]
		ttk::treeview $f.style.tree -padding 0 -height 4 \
		-show {} -columns 1 -displaycolumns 1
		$f.style.tree column #1 -width 70
		$f.style.tree insert {} end -id keyword -values {Keyword}
		$f.style.tree insert {} end -id type    -values {DataType}
		$f.style.tree insert {} end -id quote   -values {Quoted}
		$f.style.tree insert {} end -id comment -values {Comment}
		ttk::frame $f.style.c
		ttk::label $f.style.c.fgl -text FG: -pad {2 0}
		ttk::label $f.style.c.bgl -text BG: -pad {2 0}
		label $f.style.c.fg -width 6 -bg black 
		label $f.style.c.bg -width 6 -bg white
		foreach _w [list $f.style.c.fg $f.style.c.bg] {
			bind $_w <1> [namespace code [subst -nocommands {
				set _col [tk_chooseColor -parent $root -initialcolor [%W cget -bg]]
				if {\$_col ne {}} {
					%W configure -bg \$_col
					updateEditorStyle [$f.style.tree selection]
				}
			}]]
		}

		ttk::frame $f.style.s
		ttk::checkbutton $f.style.s.bold -text [msgcat::mc Bold]
		ttk::checkbutton $f.style.s.underline -text [msgcat::mc Underline]
		ttk::checkbutton $f.style.s.italic -text [msgcat::mc Italic]
		ttk::checkbutton $f.style.uppercase -text [msgcat::mc Uppercase]
		foreach _w [list $f.style.s.bold $f.style.s.underline $f.style.s.italic $f.style.uppercase] {
			$_w configure -command [subst {
				after 0 [namespace current]::updateEditorStyle \[$f.style.tree selection\]
			}]
		}
		text $f.style.sample -height 0 -relief flat -state disabled

		pack $f.style.tree -side left
		pack $f.style.c -fill x
		pack $f.style.c.fgl $f.style.c.fg $f.style.c.bgl $f.style.c.bg -side left
		pack $f.style.s -side top -fill x
		pack $f.style.s.bold $f.style.s.underline $f.style.s.italic -side left
		pack $f.style.uppercase -side top -fill x
		pack $f.style -fill x
		pack $f.style.sample -side top -fill both -expand yes -padx 4 -pady 4
		
		$f.style.sample tag configure keyword
		$f.style.sample tag configure type
		$f.style.sample tag configure quote
		$f.style.sample tag configure comment
		# setup GUI from style data 
		proc setEditorStyleGUI {{tagName ""}} [subst -nocommands {
			variable info
			if {\$tagName eq ""} return
			set d [set info(sqltext,style,[set tagName])]
			$f.style.c.fg configure -bg [dict get \$d fg]
			$f.style.c.bg configure -bg [dict get \$d bg]
			set ::$f.style.s.bold       [dict get \$d bold]
			set ::$f.style.s.underline  [dict get \$d underline]
			set ::$f.style.s.italic     [dict get \$d italic]
			set ::$f.style.uppercase    [dict get \$d uppercase]
			updateEditorSample \$tagName
		}]
		# set style data from GUI
		proc updateEditorStyle {{tagName ""}} [subst -nocommands {
			variable info
			if {\$tagName eq ""} return
			set d [dict create]
			dict set d fg [$f.style.c.fg cget -bg]
			dict set d bg [$f.style.c.bg cget -bg]
			dict set d bold      [$f.style.s.bold      instate selected]
			dict set d underline [$f.style.s.underline instate selected]
			dict set d italic    [$f.style.s.italic    instate selected]
			dict set d uppercase [$f.style.uppercase   instate selected]
			set info(sqltext,style,\$tagName) \$d
			updateEditorSample \$tagName
		}]
		# update Sampe from style data
		proc updateEditorSample {{tagName ""}} [subst -nocommands {
			variable info
			if {\$tagName eq ""} return
			set d [set info(sqltext,style,\$tagName)]

			set font [$f.style.sample cget -font]
			if {[lsearch [font names] \$font] > -1} {
				set font [list \
				[font configure \$font -family] \
				[font configure \$font -size]]
			}
			if {[dict get \$d bold]} {lappend font bold}
			if {[dict get \$d underline]} {lappend font underline}
			if {[dict get \$d italic]} {lappend font italic}
			$f.style.sample tag configure \$tagName \
				-background [dict get \$d bg] \
				-foreground [dict get \$d fg] \
				-font \$font
			set str ""
			switch -- [set tagName] {
				keyword {set str "Keyword Sample"}
				type    {set str "DataType Sample"}
				quote   {set str "'Quoted Sample'"}
				comment {set str "/*Comment Sample*/"}
			}
			if {[dict get \$d uppercase]} {
				set str [string toupper [set str]]
			}
			$f.style.sample configure -state normal
			$f.style.sample delete 0.0 end
			$f.style.sample insert end [set str] [set tagName]
			$f.style.sample configure -state disabled
			return
		}]
		bind $f.style.tree <<TreeviewSelect>> [namespace code {
			setEditorStyleGUI [%W selection]
		}]
		foreach _w [list $f.font.name $f.font.size] {
		bind $_w <<ComboboxSelected>> [namespace code [subst -nocommands {
			$f.style.sample configure -font [list \$info(sqltext,fontname) \$info(sqltext,fontsize)]
			updateEditorSample [$f.style.tree selection]
		}]]
		}

		# Notebook/Encoding
		#----------------------------------------------------------------
		set f [ttk::frame $note.encoding -padding 4]
		$note add $f -text [msgcat::mc Encoding] -sticky news

		ttk::button $f.addall -text addAll
		ttk::button $f.del    -text delete
		ttk::button $f.delall -text deleteAll
		ttk::button $f.add -text add
		foreach w [winfo children $f] {
			Util::bindMouseEvent $w
		}
		set enctree [ttk::treeview $f.tree -padding 0 -height 30 \
			-show tree -yscrollcommand "$f.y set"]
		ttk::scrollbar $f.y -command "$enctree yview"
		ttk::combobox $f.ecombo -state readonly
		
		grid $enctree  -column 0 -row 0 -rowspan 4 -sticky news
		grid $f.y      -column 1 -row 0 -rowspan 4 -sticky ns
		grid $f.ecombo -column 0 -row 4 -columnspan 2 -sticky ew
		grid $f.addall -column 2 -row 0 -pady 2 -padx 4 -sticky ew
		grid $f.del    -column 2 -row 1 -pady 2 -padx 4 -sticky ew
		grid $f.delall -column 2 -row 2 -pady 2 -padx 4 -sticky ew
		grid $f.add    -column 2 -row 4 -pady 2 -padx 4 -sticky ew
		grid rowconfigure $f 3 -weight 1
		grid columnconfigure $f 0 -weight 1
		
		$f.ecombo configure -postcommand \
		"$f.ecombo configure -values \$ENCODING_NAMES"

		$f.addall configure -command \
		"$enctree delete \[$enctree children {}\]
		foreach _enc \$ENCODING_NAMES {
			$enctree insert {} end -id \$_enc -text \$_enc
		}; unset _enc"

		$f.del configure -command \
		"$enctree delete \[$enctree selection\]"

		$f.delall configure -command \
		"$enctree delete \[$enctree children {}\]"

		$f.add configure -command \
		"set _enclist \[concat \[$enctree children {}\] \[$f.ecombo get\]]
		$enctree delete \[$enctree children {}\]
		if {\[llength \$_enclist\] == 0} break
		foreach _enc \[lsort -dictionary -uniq \$_enclist\] {
			$enctree insert {} end -id \$_enc -text \$_enc
		}; unset _enc; unset _enclist;"

		# Notebook/SQLite
		#----------------------------------------------------------------
		set f [ttk::frame $note.sqlite -padding 4]
		$note add $f -text "SQLite" -sticky news

		ttk::checkbutton $f.legacyfmt \
			-variable [namespace current]::info(sqlite,v3,legacyfileformat) \
			-text [msgcat::mc "Create/Vacuum 3.x File as 3.x Compatibility Format"]
		pack $f.legacyfmt -fill x -pady 2

		ttk::labelframe $f.v3 -padding 2 -text [msgcat::mc "Version 3.x File Creation"]

		ttk::label $f.v3.pagesizel -text "[msgcat::mc {Page Size}] : "
		ttk::combobox $f.v3.pagesize -state readonly -width 8\
			-values {512 1024 2048 4096 8192} \
			-textvariable [namespace current]::info(sqlite,v3,pagesize)
		ttk::label $f.v3.encodingl -text "[msgcat::mc {Encoding}] : "
		ttk::combobox $f.v3.encoding -state readonly -width 8\
			-textvariable [namespace current]::info(sqlite,v3,encoding)\
			-values {UTF-8 UTF-16le UTF-16be}
		pack $f.v3 -fill x 

		grid $f.v3.pagesizel -row 1 -column 0 -pady 2 -padx 2 -sticky e
		grid $f.v3.pagesize  -row 1 -column 1 -pady 2 -padx 2
		grid $f.v3.encodingl -row 1 -column 2 -pady 2 -padx 2 -sticky e
		grid $f.v3.encoding  -row 1 -column 3 -pady 2 -padx 2
		grid columnconfigure $f.v3 2 -weight 1 
		
		ttk::labelframe $f.ext -padding 2 \
			-text [msgcat::mc "Tcl Script / Shared Library"]

 		ttk::label $f.ext.detail -justify left -background white -relief groove \
		-text [msgcat::mc "NOTE:They'll be loaded when db is opened."]

		ttk::button $f.ext.add    -text [msgcat::mc "Add"]
		ttk::button $f.ext.delete -text [msgcat::mc "Remove"]
		ttk::button $f.ext.config -text [msgcat::mc "Config"]
		ttk::button $f.ext.up     -text [msgcat::mc "Up"]
		ttk::button $f.ext.down   -text [msgcat::mc "Down"]
		Util::bindMouseEvent $f.ext.add
		Util::bindMouseEvent $f.ext.delete
		Util::bindMouseEvent $f.ext.config
		Util::bindMouseEvent $f.ext.up
		Util::bindMouseEvent $f.ext.down

		::Cmd::createTreectrl $f.ext.list both
		set tree $f.ext.list.tlist
		$tree configure -height 60
		$tree column create -width 20  -tag ENABLE     -text Enable
		$tree column create -width 60  -tag NAME       -text Name
		$tree column create -width 60  -tag ENTRYPOINT -text EntryPoint
		$tree column create -width 200 -tag PATH       -text Path
		::Cmd::changeTreectrlHeaderLoolAndFeele $tree
		$tree notify bind $tree <Selection> \
		"::Cmd::changeSelectionStyleInTreectrl $tree \
		ENABLE eWin -window CheckInTreeSelected CheckInTree"
		# add/remove item procs.
		proc SqliteExt_addTreeItem {tree enable path entrypoint} {
			set check [ttk::checkbutton $tree.check[clock clicks] -style CheckInTree]
			bind $check <Destroy> {catch {unset %W}}
			uplevel #0 set $check "$enable"
			set item [$tree item create]
			$tree item style set $item ENABLE     STYLE_WIN
			$tree item style set $item NAME       STYLE_TEXT
			$tree item style set $item ENTRYPOINT STYLE_TEXT
			$tree item style set $item PATH       STYLE_TEXT
			$tree item element configure $item ENABLE eWin -window $check
			$tree item text $item NAME       [file tail $path]
			$tree item text $item ENTRYPOINT $entrypoint
			$tree item text $item PATH       $path
			$tree item lastchild root $item
		}
		proc SqliteExt_deleteTreeItem {tree} {
			foreach item [$tree selection get] {
				$tree item delete $item
			}
		}
		proc SqliteExt_add {} {
			variable info
			variable sqlite_ext_tree

			if {$info(sqlite_ext_dir) eq ""} {
				set info(sqlite_ext_dir) [pwd]
			}
			set file [tk_getOpenFile \
				-parent $sqlite_ext_tree \
				-initialdir $info(sqlite_ext_dir) \
				-filetypes $::pref(openTypeExtension)]
			if {[file exists $file]} {
				SqliteExt_addTreeItem $sqlite_ext_tree 1 $file ""
				set info(sqlite_ext_dir) [file dir $file]
			}
		}
		proc SqliteExt_remove {} {
			variable sqlite_ext_tree
			SqliteExt_deleteTreeItem $sqlite_ext_tree
		}
		proc SqliteExt_modify {tree item} {
			set title [msgcat::mc "Configure entry point of sqlite extension"]
			set msg [msgcat::mc "Enter entry point."]\n[msgcat::mc "If the value is empty, default entry point is used for shared library. And if the library is a Tcl script, the value is ignored."]\n[$tree item text $item PATH]
			set val [::GetString::show $title $msg [$tree item text $item ENTRYPOINT] $tree]
			if {$::GetString::state ne "ok"} {return}
			$tree item text $item ENTRYPOINT $val
		}
		proc SqliteExt_config {} {
			variable sqlite_ext_tree
			set tree $sqlite_ext_tree
			set sel [$tree selection get]
			if {[llength $sel] > 0} {
				SqliteExt_modify $tree [lindex $sel 0]
			}
		}
		proc SqliteExt_move {dir} {
			variable sqlite_ext_tree
			set tree $sqlite_ext_tree
			set src [lindex [$tree selection get] 0]
			if {$src eq ""} {return}
			switch -exact -- $dir {
				up {set cmd prevsibling}
				down {set cmd nextsibling}
				default {return}
			}
			set target [$tree item $cmd $src]
			if {$target eq ""} {return}
			$tree item $cmd $target $src
		}
		proc SqliteExt_bindTreeDouble1 {tree x y} {
			set id [$tree identify $x $y]
			if {$id eq ""} {return}
			if {[lindex $id 0] eq "item"} {
				SqliteExt_modify $tree [lindex $id 1]
			}
		}
		proc SqliteExt_bindTreeKeyReturn {tree} {
			set sel [$tree selection get]
			if {[llength $sel] > 0} {
				SqliteExt_modify $tree [lindex $sel 0]
			}
		}
		bind $tree <Key-Return> [namespace code {SqliteExt_bindTreeKeyReturn %W}]
		bind $tree <Double-1> [namespace code {SqliteExt_bindTreeDouble1 %W %x %y}]
		$f.ext.add configure -command [namespace code SqliteExt_add]
		$f.ext.delete configure -command [namespace code SqliteExt_remove]
		$f.ext.config configure -command [namespace code SqliteExt_config]
		$f.ext.up   configure -command [namespace code {SqliteExt_move up}]
		$f.ext.down configure -command [namespace code {SqliteExt_move down}]

		set sqlite_ext_tree $tree

		pack $f.ext.detail -side top -fill x -padx 2 -pady {0 2}
		pack $f.ext.list -side left -fill both -expand yes -padx 2
		pack $f.ext.add    -side top -pady {0 4} -padx 2
		pack $f.ext.config -side top -pady {0 4} -padx 2
		pack $f.ext.delete -side top -pady {0 4} -padx 2
		pack $f.ext.up     -side top -pady {0 4} -padx 2
		pack $f.ext.down   -side top -pady {0 0} -padx 2

		pack $f.ext -fill both -expand yes -padx 2 -pady 2
	}
	
	# reset encoding table
	$enctree delete [$enctree children {}]
	foreach enc [lsort -dictionary -uniq $::pref(enable_encoding)] {
		if {[lsearch -exact $::ENCODING_NAMES $enc] < 0} {continue}
		$enctree insert {} end -id $enc -text $enc
	}
	# reset sqlite extension table
	$sqlite_ext_tree item delete all
	foreach {enable extpath entrypoint} $info(sqlite,extension) {
		SqliteExt_addTreeItem $sqlite_ext_tree $enable $extpath $entrypoint
	}

	wm geometry $root [join $iniSize x]
	eval wm minsize $root $iniSize
	return
}

# set to preference
proc GUICmd::Preference::setVal {} {;#<<<
	variable enctree
	variable info

	# Encoding
	# init in _init proc

	# Editor
	set info(sqltext,fontname)  [lindex $SQLText::info(font) 0]
	set info(sqltext,fontsize)  [lindex $SQLText::info(font) 1]
	set info(sqltext,style,keyword) $SQLText::info(style,keyword)
	set info(sqltext,style,type)    $SQLText::info(style,type)
	set info(sqltext,style,quote)   $SQLText::info(style,quote)
	set info(sqltext,style,comment) $SQLText::info(style,comment)

	# Table
	set info(table,vi) $Table::info(vibind)

	# History
	set info(history,max) $History::info(max)

	# SQLite
	set info(sqlite,v3,legacyfileformat) $Sqlite::info(v3,legacyfileformat)
	set info(sqlite,v3,pagesize)         $Sqlite::info(v3,pagesize)
	set info(sqlite,v3,encoding)         $Sqlite::info(v3,encoding)
	set info(sqlite,extension)           $Sqlite::info(extension)
};#>>>
# get from preference
proc GUICmd::Preference::getVal {} {;#<<<
	variable enctree
	variable sqlite_ext_tree
	variable info
	
	# Encoding
	set ::pref(enable_encoding) [lsort -dictionary -uniq [$enctree children {}]]
	
	# Editor
	set SQLText::info(font) [list $info(sqltext,fontname) $info(sqltext,fontsize)]
	set SQLText::info(style,keyword) $info(sqltext,style,keyword)
	set SQLText::info(style,type)    $info(sqltext,style,type)
	set SQLText::info(style,quote)   $info(sqltext,style,quote)
	set SQLText::info(style,comment) $info(sqltext,style,comment)
	
	# Table
	set Table::info(vibind) $info(table,vi)
	
	# History
	if {[string is digit $info(history,max)]} {
		set num [string trimleft $info(history,max) 0]
		if {$num eq {}} {set num 0}
		set History::info(max) $num
	}

	# SQLite
	set Sqlite::info(v3,legacyfileformat) $info(sqlite,v3,legacyfileformat)
	set Sqlite::info(v3,pagesize)         $info(sqlite,v3,pagesize)
	set Sqlite::info(v3,encoding)         $info(sqlite,v3,encoding)
	set temp ""
	foreach item [$sqlite_ext_tree item children root] {
		set win [$sqlite_ext_tree item element cget $item ENABLE eWin -window]
		lappend temp [$win instate selected]
		lappend temp [$sqlite_ext_tree item text $item PATH]
		lappend temp [$sqlite_ext_tree item text $item ENTRYPOINT]
	}
	set Sqlite::info(extension)           $temp
	Sqlite::updateInfo
};#>>>

;#>>>
# export Text File
namespace eval GUICmd::ExportText {;#<<<
	# widget
	variable root .cmdExportText
	variable dcombo {}
	variable tcombo {}
	# var
	variable info
	array set info {
		src         result
		db          {}
		table       {}
		sep         ","
		sepcustom   "|"
		encoding	{}
		return      excel
		title       1
		directory   {}
	}
	variable iniSize {300 240}
}

proc GUICmd::ExportText::run {} {;#<<<
	variable info
	variable iniSize
	variable state
	variable root
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	
	_init

	# Init 
	wm transient $root .
	::tk::SetFocusGrab $root $root
	if {[winfo width $root] == 1} {
		::Util::moveCenter $root $iniSize
	} else {
		::Util::moveCenter $root
	}
	wm deiconify $root
	focus -force $root
	raise $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state ne "ok"} { return }

	# Get file name
	set file [tk_getSaveFile -initialdir $info(directory) -filetypes $::pref(openTypeText)]
	if {$file eq {}} return
	
	set info(directory) [file dir $file]
	export $file
;#    Cmd::updateFrames
};#>>>

proc GUICmd::ExportText::_init {} {;#<<<
	variable info
	variable root
	variable iniSize
	variable dcombo
	variable tcombo

	if {![winfo exists $root]} {
		set root [toplevel $root]
		wm withdraw $root
		set f [ttk::frame $root.f -padding 4]
		pack $f -expand yes -fill both
		wm title $root [msgcat::mc "Export Text Format"]

		# OK / Cancel Button
		ttk::frame $f.cmd
		ttk::button $f.cmd.ok     -text OK
		ttk::button $f.cmd.cancel -text Cancel
		foreach w [winfo children $f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $f.cmd -fill x -side bottom -pady 4
		pack $f.cmd.cancel $f.cmd.ok -side right -padx 2
		
		# Root binding
		set cancelAction [namespace code {wm withdraw $root;set state cancel}]
		set okAction [namespace code {wm withdraw $root;set state ok}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$f.cmd.ok     configure -command $okAction
        $f.cmd.cancel configure -command $cancelAction

		# Options
		ttk::frame $f.opt
		ttk::labelframe $f.opt.in -text [msgcat::mc Source] -padding 4 -takefocus 0
		ttk::radiobutton $f.opt.in.result -text [msgcat::mc "Result Table"] -value result -variable [namespace current]::info(src)
		ttk::radiobutton $f.opt.in.table  -text [msgcat::mc Table] -value table  -variable [namespace current]::info(src)
		ttk::label $f.opt.in.ldb -text [msgcat::mc Database] -takefocus 0
		set dcombo [ttk::combobox $f.opt.in.db -state read -textvariable [namespace current]::info(db)]
		ttk::label $f.opt.in.ltbl -text [msgcat::mc Table] -takefocus 0
		set tcombo [ttk::combobox $f.opt.in.tbl  -state read -textvariable [namespace current]::info(table)]
		
		grid $f.opt.in.result -row 0 -column 0 -columnspan 2 -sticky ew
		grid $f.opt.in.table  -row 1 -column 0 -columnspan 2 -sticky ew
		grid $f.opt.in.ldb  -row 2 -column 0 -padx {30 5} -sticky w
		grid $f.opt.in.db   -row 2 -column 1 -sticky ew -pady 2
		grid $f.opt.in.ltbl -row 3 -column 0 -padx {30 5} -sticky w
		grid $f.opt.in.tbl  -row 3 -column 1 -sticky ew -pady 2
		grid columnconfigure $f.opt.in 1 -weight 1

		$f.opt.in.result configure -command \
		" foreach _w \{ $f.opt.in.ldb $f.opt.in.db $f.opt.in.ltbl $f.opt.in.tbl \} {
			\$_w configure -state disabled
		} "
		$f.opt.in.table configure -command \
		" foreach _w \{ $f.opt.in.ldb $f.opt.in.db $f.opt.in.ltbl $f.opt.in.tbl \} {
			\$_w configure -state read
		} "
		foreach _w [list $f.opt.in.ldb $f.opt.in.db $f.opt.in.ltbl $f.opt.in.tbl] {
			$_w configure -state disabled
		}
		$f.opt.in.db configure -postcommand [namespace code {set info(pre_db) $info(db)}]
		bind $f.opt.in.db <<ComboboxSelected>> [namespace code {
			if {$info(db) ne $info(pre_db)} {
				initTableList
			}
		}]
		proc initDbList args {
			variable dcombo
			$dcombo configure -values [Sqlite::getDBList all]
			$dcombo current 0
			initTableList
		}
		proc initTableList args {
			variable tcombo
			variable info
			set info(table) ""
			set tlist [list]
			if {$info(db) eq ""} {
				set tlist [list]
			} else {
				set tlist [lsort -dictionary [Sqlite::getTableList $info(db)]]
				set vlist [lsort -dictionary [Sqlite::getViewList $info(db)]]
				if {[llength $vlist] > 0} {
					lappend tlist $vlist
				}
			}
			$tcombo configure -values $tlist
			if {[llength $tlist] > 0} {
				$tcombo current 0
			}
		}	

		ttk::label $f.opt.lsep -text [msgcat::mc Separator] -takefocus 0
		ttk::radiobutton $f.opt.sepComma -text [msgcat::mc Comma] -value "," \
			-variable [namespace current]::info(sep)
		ttk::radiobutton $f.opt.sepTab   -text [msgcat::mc Tab] -value "\t" \
			-variable [namespace current]::info(sep)
		ttk::radiobutton $f.opt.sepCustom -text [msgcat::mc Custom] -value custom \
			-variable [namespace current]::info(sep)
		ttk::entry $f.opt.sepCustomValue -width 3 -state disabled \
			-textvariable [namespace current]::info(sepcustom)
		ttk::label $f.opt.lret -text [msgcat::mc "Record Type"] -takefocus 0
		ttk::radiobutton $f.opt.retAuto  -text [msgcat::mc Auto] -value auto -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retCRLF -text CRLF(Win) -value crlf -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retLF   -text LF(Unix) -value lf -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retExcel -text Excel -value excel  -variable [namespace current]::info(return)
		ttk::label $f.opt.lenc -text [msgcat::mc "Output Encoding"] -takefocus 0
		ttk::combobox $f.opt.enc -state read -textvariable [namespace current]::info(encoding) \
			-postcommand "$f.opt.enc configure -values \$::pref(enable_encoding)"
		ttk::checkbutton $f.opt.ttl -text [msgcat::mc "Output Field Names"] -variable [namespace current]::info(title)
		foreach w [winfo children $f.opt] {
			Util::bindMouseEvent $w
		}

		$f.opt.sepComma  configure -command "$f.opt.sepCustomValue state disabled"
		$f.opt.sepTab    configure -command "$f.opt.sepCustomValue state disabled"
		$f.opt.sepCustom configure -command "$f.opt.sepCustomValue state !disabled"
		Util::setOneCharValidateToTtkentry $f.opt.sepCustomValue

		grid $f.opt.in   -row 0 -column 0 -columnspan 5 -sticky news
		grid $f.opt.lsep     -row 1 -column 0  -sticky w
		grid $f.opt.sepComma -row 1 -column 1  -sticky ew
		grid $f.opt.sepTab   -row 1 -column 2  -sticky ew
		grid $f.opt.sepCustom      -row 1 -column 3 -sticky ew
		grid $f.opt.sepCustomValue -row 1 -column 4 -sticky w
		grid $f.opt.lret     -row 2 -column 0  -sticky w
		grid $f.opt.retAuto  -row 2 -column 1  -sticky ew
		grid $f.opt.retCRLF  -row 2 -column 2  -sticky ew
		grid $f.opt.retLF    -row 2 -column 3  -sticky ew
		grid $f.opt.retExcel -row 2 -column 4  -sticky ew
		grid $f.opt.lenc     -row 3 -column 0  -sticky w
		grid $f.opt.enc      -row 3 -column 1 -columnspan 2 -sticky ew
        grid $f.opt.ttl  -row 4 -column 0 -columnspan 4 -sticky ew
		foreach n {0 1 2 3 4} { grid columnconfigure $f.opt $n -weight 1}
		pack $f.opt -fill both -side top

		# root size
		::Util::updateWindowSize $root
;#        wm geometry $root [join $iniSize x]
;#        eval wm minsize $root $iniSize
		wm resizable $root 0 0
	}
	initDbList
	set info(encoding) $::database(encoding)
};#>>>

proc GUICmd::ExportText::export {file} {;#<<<
	variable info
	set dlist [list]
	switch -exact -- $info(src) {
		result {
			Table::getList rows cols dlist
			incr cols -1
			incr rows -1
		}
		table {
			set q "SELECT * FROM '$info(db)'.'$info(table)' WHERE rowid IS NULL"
			if {[Sqlite::getColumnNamesFromEval $q] != 0} return
			set dlist $Sqlite::data
			set cols [expr {[llength $dlist] -1}]
			set q "SELECT * FROM '$info(db)'.'$info(table)'"
			if {[Sqlite::evalQuery $q] != 0} return
			set dlist [concat $dlist $Sqlite::data]
		}
	}
	if !$info(title) {
		set dlist [lrange $dlist [expr {$cols+1}] end]
	}
	
	set fp [open $file w]
	
	set map [list \" \"\"]
	set ret \n
	switch -exact -- $info(return) {
		auto { fconfigure $fp -translation auto -encoding $info(encoding) }
		crlf { fconfigure $fp -translation crlf -encoding $info(encoding) }
		lf   { fconfigure $fp -translation lf   -encoding $info(encoding) }
		excel {
			set map [list \r\n \n \r \n \" \"\"]
			fconfigure $fp -translation binary -encoding $info(encoding)
			set ret \r\n
		}
	}

	if {$info(encoding) eq "unicode"} {
		::Util::Unicode::writeUTF16BOM $fp
	}
	
	set i 0
	set sep $info(sep)
	if {$sep eq "custom"} {set sep $info(sepcustom)}
	foreach d $dlist {
		puts -nonewline $fp \"[string map $map $d]\"
		if {$i < $cols} {
			puts -nonewline $fp $sep
			incr i
		} else {
			puts -nonewline $fp $ret
			set i 0
		}
	}
	close $fp
};#>>>
;#>>>
# import Text File
namespace eval GUICmd::ImportText {;#<<<
	# widget
	variable root .cmdImportText
	variable dcombo {}
	variable tcombo {}
	# var
	variable info
	array set info {
		dst         new
		db          {}
		newtable    {}
		table       {}
		sep         ","
		sepcustom   "|"
		quote       "\""
		quotecustom "`"
		encoding	{}
		return      excel
		title       1
		file        {}
		directory   {}
	}
	variable iniSize {300 240}
}

proc GUICmd::ImportText::run {} {
	variable info
	variable iniSize
	variable state
	variable root
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}

	# Get file name
	set info(file) [tk_getOpenFile -initialdir $info(directory) -filetypes $::pref(openTypeText)]
	if {$info(file) eq {}} return
	
	set info(directory) [file dir $info(file)]
	_init

	# Init 
	wm transient $root .
	::tk::SetFocusGrab $root $root
	if {[winfo width $root] == 1} {
		::Util::moveCenter $root $iniSize
	} else {
		::Util::moveCenter $root
	}
	wm deiconify $root
	focus -force $root
	raise $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state ne "ok"} { return }
	
	import
	Cmd::updateFrames
}

proc GUICmd::ImportText::_init {} {
	variable info
	variable root
	variable iniSize
	variable dcombo
	variable tcombo

	if {![winfo exists $root]} {
		set root [toplevel $root]
		wm withdraw $root
		set f [ttk::frame $root.f -padding 4]
		pack $f -expand yes -fill both
		wm title $root [msgcat::mc "Import Text Format"]

		# OK / Cancel Button
		ttk::frame $f.cmd
		ttk::button $f.cmd.preview -text [msgcat::mc Preview]
		ttk::button $f.cmd.ok     -text OK
		ttk::button $f.cmd.cancel -text Cancel
		foreach w [winfo children $f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $f.cmd -fill x -side bottom -pady 4
		pack $f.cmd.cancel $f.cmd.ok -side right -padx 2
		pack $f.cmd.preview -side left -padx 2
		
		# Root binding
		set cancelAction [namespace code {wm withdraw $root;set state cancel}]
		set okAction [namespace code {wm withdraw $root;set state ok}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$f.cmd.ok     configure -command $okAction
        $f.cmd.cancel configure -command $cancelAction
		$f.cmd.preview configure -command [namespace current]::preview
		
		# Options
		ttk::frame $f.opt
		ttk::labelframe $f.opt.out -text [msgcat::mc "Import to"] -padding 4 -takefocus 0
		ttk::label $f.opt.out.ldb -text [msgcat::mc Database] -takefocus 0
		set dcombo [ttk::combobox $f.opt.out.db -state read -textvariable [namespace current]::info(db)]
		ttk::radiobutton $f.opt.out.newtable -text [msgcat::mc "New Table"] -value new -variable [namespace current]::info(dst)
		ttk::label $f.opt.out.lnewname -text [msgcat::mc "New Table"] -takefocus 0
		ttk::entry $f.opt.out.newname -textvariable [namespace current]::info(newtable)
		Util::bindEditorPopCommon $f.opt.out.newname
		ttk::radiobutton $f.opt.out.table  -text [msgcat::mc Table] -value table  -variable [namespace current]::info(dst)
		ttk::label $f.opt.out.ltbl -text [msgcat::mc Table] -takefocus 0
		set tcombo [ttk::combobox $f.opt.out.tbl  -state read -textvariable [namespace current]::info(table)]
		
		grid $f.opt.out.ldb  -row 0 -column 0 -sticky w
		grid $f.opt.out.db   -row 0 -column 1 -sticky ew -pady 2 
		grid $f.opt.out.newtable -row 1 -column 0 -padx {30 5} -pady 2 -sticky ew
		grid $f.opt.out.newname   -row 1 -column 1 -sticky ew
		grid $f.opt.out.table  -row 2 -column 0 -padx {30 5} -pady 2 -sticky ew
		grid $f.opt.out.tbl  -row 2 -column 1 -sticky ew
		grid columnconfigure $f.opt.out 1 -weight 1

		$f.opt.out.newtable configure -command \
		"  $f.opt.out.tbl configure -state disabled
		   $f.opt.out.newname configure -state normal "
		$f.opt.out.table configure -command \
		"  $f.opt.out.tbl configure -state read
		   $f.opt.out.newname configure -state disabled "
		$f.opt.out.tbl configure -state disabled
		

		$f.opt.out.db configure -postcommand [namespace code {set info(pre_db) $info(db)}]
		bind $f.opt.out.db <<ComboboxSelected>> [namespace code {
			if {$info(db) ne $info(pre_db)} {
				initTableList
			}
		}]
		proc initDbList args {
			variable dcombo
			if {[Sqlite::getCurrentVersion] == 2} {
				$dcombo configure -values {main temp}
			} else {
				$dcombo configure -values [Sqlite::getDBList all]
			}
			$dcombo current 0
			initTableList
		}
		proc initTableList args {
			variable tcombo
			variable info
			set info(table) ""
			set tlist [list]
			if {$info(db) eq ""} {
				set tlist [list]
			} else {
				set tlist [lsort -dictionary [Sqlite::getTableList $info(db)]]
			}
			$tcombo configure -values $tlist
			if {[llength $tlist] > 0} {
				$tcombo current 0
			}
		}	

		ttk::label $f.opt.lsep -text [msgcat::mc Separator] -takefocus 0
		ttk::radiobutton $f.opt.sepComma -text [msgcat::mc Comma] -value "," \
			-variable [namespace current]::info(sep)
		ttk::radiobutton $f.opt.sepTab   -text [msgcat::mc Tab] -value "\t" \
			-variable [namespace current]::info(sep)
		ttk::radiobutton $f.opt.sepCustom -text [msgcat::mc Custom] -value custom \
			-variable [namespace current]::info(sep)
		ttk::entry $f.opt.sepCustomValue -width 3 -state disabled \
			-textvariable [namespace current]::info(sepcustom)
		ttk::label $f.opt.lret -text [msgcat::mc "Record Type"] -takefocus 0
		ttk::radiobutton $f.opt.retAuto  -text [msgcat::mc Auto] -value auto -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retCRLF -text CRLF(Win) -value crlf -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retLF   -text LF(Unix) -value lf -variable [namespace current]::info(return)
		ttk::radiobutton $f.opt.retExcel -text Excel -value excel  -variable [namespace current]::info(return)
		ttk::label $f.opt.lquote -text [msgcat::mc "Quote char"] -takefocus 0
		ttk::radiobutton $f.opt.quoteDouble -text [msgcat::mc "Double Quote"](\") \
			-value "\""  -variable [namespace current]::info(quote)
		ttk::radiobutton $f.opt.quoteSingle -text [msgcat::mc "Single Quote"](') \
			-value "'"  -variable [namespace current]::info(quote)
		ttk::radiobutton $f.opt.quoteCustom -text [msgcat::mc "Custom"] \
			-value "custom" -variable [namespace current]::info(quote)
		ttk::entry $f.opt.quoteCustomValue -width 3 -state disabled \
			-textvariable [namespace current]::info(quotecustom)
		ttk::label $f.opt.lenc -text [msgcat::mc "Input Encoding"] -takefocus 0
		ttk::combobox $f.opt.enc -state read -textvariable [namespace current]::info(encoding) \
			-postcommand "$f.opt.enc configure -values \$::pref(enable_encoding)"
		ttk::checkbutton $f.opt.ttl -text [msgcat::mc "First Line is Field Names"] -variable [namespace current]::info(title)
		foreach w [winfo children $f.opt] {
			Util::bindMouseEvent $w
		}

		
		$f.opt.sepComma  configure -command "$f.opt.sepCustomValue state disabled"
		$f.opt.sepTab    configure -command "$f.opt.sepCustomValue state disabled"
		$f.opt.sepCustom configure -command "$f.opt.sepCustomValue state !disabled"
		Util::setOneCharValidateToTtkentry $f.opt.sepCustomValue
		$f.opt.quoteSingle configure -command "$f.opt.quoteCustomValue state disabled"
		$f.opt.quoteDouble configure -command "$f.opt.quoteCustomValue state disabled"
		$f.opt.quoteCustom configure -command "$f.opt.quoteCustomValue state !disabled"
		Util::setOneCharValidateToTtkentry $f.opt.quoteCustomValue

		grid $f.opt.out   -row 0 -column 0 -columnspan 5 -sticky news
		grid $f.opt.lsep     -row 1 -column 0  -sticky w
		grid $f.opt.sepComma -row 1 -column 1  -sticky ew
		grid $f.opt.sepTab   -row 1 -column 2  -sticky ew
		grid $f.opt.sepCustom      -row 1 -column 3 -sticky ew
		grid $f.opt.sepCustomValue -row 1 -column 4 -sticky w
		grid $f.opt.lret     -row 2 -column 0  -sticky w
		grid $f.opt.retAuto  -row 2 -column 1  -sticky ew
		grid $f.opt.retCRLF  -row 2 -column 2  -sticky ew
		grid $f.opt.retLF    -row 2 -column 3  -sticky ew
		grid $f.opt.retExcel -row 2 -column 4  -sticky ew

		grid $f.opt.lquote      -row 3 -column 0 -sticky w
		grid $f.opt.quoteDouble -row 3 -column 1 -columnspan 2 -sticky ew
		grid $f.opt.quoteSingle -row 3 -column 3 -columnspan 2 -sticky ew
		grid $f.opt.quoteCustom      -row 4 -column 1 -sticky ew
		grid $f.opt.quoteCustomValue -row 4 -column 2 -sticky w

		grid $f.opt.lenc     -row 5 -column 0  -sticky e
		grid $f.opt.enc      -row 5 -column 1 -columnspan 2 -sticky ew
        grid $f.opt.ttl  -row 6 -column 0 -columnspan 3 -sticky ew
		foreach n {0 1 2 3 4} { grid columnconfigure $f.opt $n -weight 1}
		pack $f.opt -fill both -side top

		# root size
		::Util::updateWindowSize $root
;#        wm geometry $root [join $iniSize x]
;#        eval wm minsize $root $iniSize
		wm resizable $root 0 0
	}
	initDbList
	set info(encoding) $::database(encoding)
}

proc GUICmd::ImportText::preview {} {
	variable info
	variable root
	set width 400
	set height 250
	# create gui
	set pre .cmdImportTextPreview
	if {[winfo exists $pre]} {
		destroy $pre
	}
	toplevel $pre
	set f [ttk::frame $pre.f -padding 4]
	pack $f -expand yes -fill both
	wm title $pre [msgcat::mc Preview]

	# OK / Cancel Button
	ttk::frame $f.cmd
	ttk::button $f.cmd.close -text [msgcat::mc Close]
	foreach w [winfo children $f.cmd] {
		Util::bindMouseEvent $w
	}
	ttk::label $f.note -wraplength [expr $width -20] -justify left -background white -relief groove \
	-text [msgcat::mc "Note:If import file size was big, we would preview top 50KB of the file."]
	
	pack $f.cmd -fill x -side bottom -pady 4
	pack $f.note -fill x -side top -padx 2 -pady 4
	pack $f.cmd.close -side right -padx 2
	
	ttk::frame $f.bd -style TEntry -border 2
	table $f.bd.table \
		-cache 1 \
		-rows 0 -cols 0 \
		-selectmode extended \
		-titlerows 1 -titlecols 1 -selecttitle 0 \
		-xscrollcommand "$f.bd.x set" \
		-yscrollcommand "$f.bd.y set"
	ttk::scrollbar $f.bd.y -command "$f.bd.table yview"
	ttk::scrollbar $f.bd.x -command "$f.bd.table xview" -orient horizontal
	ttk::frame $f.bd.bg
	grid $f.bd.table -row 0 -column 0 -sticky news
	grid $f.bd.y     -row 0 -column 1 -sticky ns
	grid $f.bd.x     -row 1 -column 0 -sticky ew
	grid $f.bd.bg    -row 1 -column 1 -sticky news
	grid columnconfigure $f.bd 0 -weight 1
	grid rowconfigure $f.bd 0 -weight 1
	pack $f.bd -fill both -side top -expand yes

	set tablew $f.bd.table
	$tablew width 0 3
	::Cmd::changeTableLookAndFeel $tablew
	bind $tablew <MouseWheel> {
		%W yview scroll [expr {-(%D/120) * 4}] units
	}
	set deleteCmd "destroy $pre; ::tk::RestoreFocusGrab $pre $pre destroy"
	$f.cmd.close configure -command $deleteCmd
	wm protocol $pre WM_DELETE_WINDOW $deleteCmd

	# set data

	# create temp file
	if {[catch {open $info(file) r} ofp]} {
		::Message::show -type apperror \
		-message [msgcat::mc {Failed to open the file.}] \
		-detail $info(file)
		return 
	}
	set tempname $::pref(appname)_temp_[file rootname [file tail $info(file)]]
	set tempname_orig $tempname
	set i 0
	while {[file exists [file join [file dir $info(file)] $tempname]]} {
		set tempname $tempname_orig$i
		incr i
	}
	set tempname [file exists [file join [file dir $info(file)] $tempname]]
	set tfp [open $tempname w]
	fconfigure $ofp -translation binary -encoding binary
	fconfigure $tfp -translation binary -encoding binary
	fcopy $ofp $tfp -size 50000 
	close $ofp
	close $tfp
	
	# set data from temp
	set fp [open $tempname r]
	switch -exact -- $info(return) {
		auto  { fconfigure $fp -translation auto -encoding $info(encoding) }
		crlf  { fconfigure $fp -translation crlf -encoding $info(encoding) }
		lf    { fconfigure $fp -translation lf   -encoding $info(encoding) }
		excel { fconfigure $fp -translation crlf -encoding $info(encoding) }
	}

	set isLittleEndian [::Util::Unicode::isLittleEndian $fp]

	# read firstline
	set sep $info(sep)
	set quote $info(quote)
	if {$sep eq "custom"} {set sep $info(sepcustom)}
	if {$quote eq "custom"} {set quote $info(quotecustom)}
	set trecord [Util::readCSVLine $fp $sep $quote $isLittleEndian]
	set colmax [llength $trecord]

	$tablew configure -state normal
	switch -exact -- $info(dst) {
		new {
			$tablew configure -cols [expr {$colmax+1}]
			if {!$info(title)} {
				seek $fp 0 start
				::Util::Unicode::readBOM $fp
				set trecord ""
				for {set i 0} {$i < $colmax} {incr i} {
					lappend trecord "Col$i"
				}
			}
			$tablew set row 0,1 $trecord
			set useColList [Util::range 0 [expr {$colmax -1}]]
		}
		table {
			set table "'$info(db)'.'$info(table)'"
			set q "SELECT * FROM $table WHERE rowid IS NULL"
			if {[::Sqlite::getColumnNamesFromEval $q] != 0} {
				close $fp
				file delete -force $tempname
				return
			}
			set collist $Sqlite::data
			$tablew configure -cols [expr {[llength $collist]+1}]
			set useColList [list]
			$tablew set row 0,1 $collist
			if {$info(title)} {
				foreach col $collist {
					lappend useColList [lsearch $trecord $col]
				}
			} else {
				seek $fp 0 start
				::Util::Unicode::readBOM $fp
				set useColList [Util::range 0 [expr {$colmax -1}]]
			}
		}
	}
	set r 1
	while {![eof $fp]} {
		set record [Util::readCSVLine $fp $sep $quote $isLittleEndian]
		if {[llength $record] == 0} continue
		set c 1
		$tablew configure -rows [expr {[$tablew cget -rows] +1}]
		foreach index $useColList {
			$tablew set $r,$c [lindex $record $index]
			incr c
		}
		incr r
	}
	$tablew configure -state disabled
	close $fp
	file delete -force $tempname

	wm geometry $pre ${width}x${height}
	Util::moveCenter $pre [list $width $height]
	wm transient $pre $root
	::tk::SetFocusGrab $pre $pre
}

proc GUICmd::ImportText::import {} {
	variable info
	set sep $info(sep)
	set quote $info(quote)
	if {$sep eq "custom"} {set sep $info(sepcustom)}
	if {$quote eq "custom"} {set quote $info(quotecustom)}
	
	if {[catch {open $info(file) r} fp]} {
		::Message::show -type apperror \
		-message [msgcat::mc {Failed to open the file.}] \
		-detail $info(file)
		return 
	}
	
	switch -exact -- $info(return) {
		auto  { fconfigure $fp -translation auto -encoding $info(encoding) }
		crlf  { fconfigure $fp -translation crlf -encoding $info(encoding) }
		lf    { fconfigure $fp -translation lf   -encoding $info(encoding) }
		excel { fconfigure $fp -translation crlf -encoding $info(encoding) }
	}

	set isLittleEndian [::Util::Unicode::isLittleEndian $fp]

	# read firstline
	set trecord [Util::readCSVLine $fp $sep $quote $isLittleEndian]
	set colmax [llength $trecord]

	switch -exact -- $info(dst) {
		new {
			if {[Sqlite::findTable $info(newtable)] > 0} {
				::Message::show -type apperror -textbox 0 -message \
				[msgcat::mc "Failed to create the table. A table with the same name already exists."]
				close $fp; return
			}
			set table "'$info(db)'.'$info(newtable)'"
			set collist [list]
			if {$info(title)} {
				foreach col $trecord { lappend collist "'$col' text" }
			} else {
				seek $fp 0 start
				::Util::Unicode::readBOM $fp
				for {set i 0} {$i < $colmax} {incr i} { lappend collist "Col$i text" }
			}
			if {[Sqlite::getCurrentVersion] == 2} {
				if {$info(db) == "temp"} {
				set q "create temp table '$info(newtable)' (\n"
				} else {
				set q "create table '$info(newtable)' (\n"
				}
			} else {
				set q "create table '$info(db)'.'$info(newtable)' (\n"
			}
			append q [join $collist ,\n]
			append q ");"
			if {[Sqlite::evalQuery $q] != 0} {close $fp; return}
			set useColList [Util::range 0 [expr {$colmax -1}]]
		}

		table {
			set table "'$info(db)'.'$info(table)'"
			set q "SELECT * FROM $table WHERE Rowid IS NULL"
			if {[::Sqlite::getColumnNamesFromEval $q] != 0} {
				#Error
				close $fp; return
			}
			set collist $Sqlite::data
			
			if {$info(title)} {
				set useColList [list]
				foreach col $collist {
					lappend useColList [lsearch $trecord $col]
				}
			} else {
				seek $fp 0 start
				set useColList [Util::range 0 [expr {[llength $collist]-1}]]
			}
		}
	}

	set autoincr [Sqlite::getAutoIncrColumn $info(db) $info(table) index]
	if {$autoincr eq {}} {set autoincr -1}

	if {[Sqlite::evalQuery "begin"] != 0} {close $fp; return}
	set i 0
	while {![eof $fp]} {
		set record [Util::readCSVLine $fp $sep $quote $isLittleEndian]
		if {[llength $record] == 0} continue
		set q "insert into $table values ("
		set vals [list]
		set i 0
		foreach index $useColList {
			set val '[string map {' ''} [lindex $record $index]]'
			if {$autoincr == $i} {
				if {$val eq "''"} {set val NULL}
			}
			lappend vals $val
			incr i
		}
		append q "[join $vals ,]);"
		if {[Sqlite::evalQuery $q] != 0} {
			Sqlite::evalQuery "commit"
			close $fp;
			return
		}

		incr i
		if {$i % 5000 == 0} {
			if {[Sqlite::evalQuery "commit"] != 0} {close $fp; return}
			if {[Sqlite::evalQuery "begin"] != 0} {close $fp; return}
		}
	}
	if {[Sqlite::evalQuery "commit"] != 0} {close $fp; return}

	close $fp
}

;#>>>
# set cache size
namespace eval GUICmd::SetCacheSize {;#<<<
	# widget
	variable root .cmdSetCacheSize
	# var
	variable info
	array set info {
		db       "main"
		csize    2000
		dsize    2000
		cmsize   1024
		dmsize   1024
		type "default cache"
	}
	variable iniSize {300 240}
}

proc GUICmd::SetCacheSize::run {} {;#<<<
	variable root
	variable info
	variable iniSize
	variable state

	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	
	_init
	
	# Init 
	wm transient $root .
	::tk::SetFocusGrab $root $root
	if {[winfo width $root] == 1} {
		::Util::moveCenter $root $iniSize
	} else {
		::Util::moveCenter $root
	}
	wm deiconify $root
	focus -force $root
	raise $root
	set state shown
	tkwait var [namespace current]::state
	::tk::RestoreFocusGrab $root $root withdraw
	if {$state ne "ok"} { return }
	
	setSize
	return
};#>>>

proc GUICmd::SetCacheSize::_init {} {
	variable info
	variable root
	variable iniSize

	if {![winfo exists $root]} {
		set root [toplevel $root]
		wm withdraw $root
		set f [ttk::frame $root.f -padding 4]
		pack $f -expand yes -fill both
		wm title $root [msgcat::mc "Set Cache Size"]

		# OK / Cancel Button
		ttk::frame $f.cmd
		ttk::button $f.cmd.ok     -text OK
		ttk::button $f.cmd.cancel -text Cancel
		foreach w [winfo children $f.cmd] {
			Util::bindMouseEvent $w
		}
		pack $f.cmd -fill x -side bottom -pady 4
		pack $f.cmd.cancel $f.cmd.ok -side right -padx 2
		
		# Root binding
		set cancelAction [namespace code {wm withdraw $root;set state cancel}]
		set okAction [namespace code {wm withdraw $root;set state ok}]
		bind $root <Escape> $cancelAction
		wm protocol $root WM_DELETE_WINDOW $cancelAction
		$f.cmd.ok     configure -command $okAction
        $f.cmd.cancel configure -command $cancelAction
		
		# Details
 		ttk::label $f.detail -justify left -background white -relief groove \
		-text [msgcat::mc "Cache size reverts to the default value when this DB is closed. But Default cache size value is stored in DB. Temp DB's cache size and default cache size aren't stored in DB."] -wraplength 280
		pack $f.detail -fill x

		# Options
		ttk::frame $f.opt
		ttk:::label $f.opt.dbl -text [msgcat::mc Database]
		ttk::combobox $f.opt.db -state readonly -textvariable [namespace current]::info(db)
		ttk::radiobutton $f.opt.rc -value "cache" -text [msgcat::mc "Cache size"]
		ttk::radiobutton $f.opt.rd -value "default cache" -text [msgcat::mc "Default cache size"]
		foreach _w [list $f.opt.rc $f.opt.rd] {
			$_w configure -variable [namespace current]::info(type)
		}
		ttk::entry $f.opt.tc -textvariable [namespace current]::info(csize)
		ttk::entry $f.opt.td -textvariable [namespace current]::info(dsize)
		ttk::label $f.opt.tcp -text " page "
		ttk::label $f.opt.tdp -text " page "
		ttk::label $f.opt.cms -justify right -anchor e -width 11 -text "(1000 MB)"
		ttk::label $f.opt.dms -justify left  -anchor e -width 11 -text "(0 MB)"
		foreach _w [list $f.opt.tc $f.opt.td] _w2 [list $f.opt.cms $f.opt.dms] {
			$_w configure -width 8 -validate key -validatecommand [subst {
				if {\[string is digit %S\]} {
					after 0 [namespace current]::updateMemorySize
					return 1
				} else {
					return 0
				}
			}]
			Util::bindEditorPopCommon $_w
		}
		proc incrCacheSize {type size} {
			variable info
			if {$size >= 0} { set info($type) $size }
			after 0 [namespace code {updateMemorySize}]
		}
		bind $f.opt.tc <Up>   [namespace code {incrCacheSize csize [expr {[%W get] + 200}]}]
		bind $f.opt.tc <Down> [namespace code {incrCacheSize csize [expr {[%W get] - 200}]}]
		bind $f.opt.td <Up>   [namespace code {incrCacheSize dsize [expr {[%W get] + 200}]}]
		bind $f.opt.td <Down> [namespace code {incrCacheSize dsize [expr {[%W get] - 200}]}]

		proc updateMemorySize [list \
			[list elist [list $f.opt.tc $f.opt.td]] \
			[list llist [list $f.opt.cms $f.opt.dms]]] {
			# basically this command is called with no args.
			variable info
			set pagesize [::Sqlite::pagesize get $info(db)]

			foreach w1 $elist w2 $llist {
				set size [string trimleft [$w1 get] 0]
				if {$size eq ""} {
					set num 0.00
				} else {
					set num [format %.2f [expr {(($pagesize / 1024 + 0.5) / 1024 * int($size)) }]]
				}
				$w2 configure -text "($num MB)"
			}
		}
		$f.opt.db configure  -postcommand [subst -nocommands {
			if {[Sqlite::getCurrentVersion] == 3} {
			$f.opt.db configure -values [Sqlite::getDBList writable]
			} else {
			$f.opt.db configure -values [list main]
			}
		}]
		bind $f.opt.db <<ComboboxSelected>> [namespace code {getSize $info(db); updateMemorySize}]

		grid $f.opt.dbl -row 0 -column 0 -sticky nws -padx {0 10}
		grid $f.opt.db  -row 0 -column 1 -sticky ew -columnspan 3 -pady {4 4}
		grid $f.opt.rc  -row 1 -column 0 -sticky ew -padx {0 10}
		grid $f.opt.tc  -row 1 -column 1 -sticky ew
		grid $f.opt.tcp -row 1 -column 2 -sticky ew -pady {4 2}
		grid $f.opt.cms -row 1 -column 3 -sticky ew
		grid $f.opt.rd  -row 2 -column 0 -sticky ew -padx {0 10}
		grid $f.opt.td  -row 2 -column 1 -sticky ew
		grid $f.opt.tdp -row 2 -column 2 -sticky ew -pady 2
		grid $f.opt.dms -row 2 -column 3 -sticky ew
		pack $f.opt -fill both -side top
		
		proc changeRadio args [subst -nocommands {
			variable info
			foreach _w [list $f.opt.tc $f.opt.tcp $f.opt.td $f.opt.tdp $f.opt.cms $f.opt.dms] {
				[set _w] configure -state disabled
			}
			if {[set info(type)] eq "cache"} {
				$f.opt.tc  configure -state normal
				$f.opt.tcp configure -state normal
				$f.opt.cms configure -state normal
			} else {
				$f.opt.td  configure -state normal
				$f.opt.tdp configure -state normal
				$f.opt.dms configure -state normal
			}
		}]
		trace add variable [namespace current]::info(type) write [namespace current]::changeRadio

		::Util::updateWindowSize $root
 		$f.detail configure -wraplength [expr {[winfo reqwidth $f] - 20}]
		wm resizable $root 0 0
	}
	getSize $info(db)
	updateMemorySize
	changeRadio
}

proc GUICmd::SetCacheSize::getSize {{db ""}} {
	variable info
	if {$db eq ""} {set db "main"}
	set info(csize) [::Sqlite::cache get "cache" $db]
	set info(dsize) [::Sqlite::cache get "default cache" $db]
}

proc GUICmd::SetCacheSize::setSize {} {
	variable info
	if {$info(db) eq ""} {set $info(db) "main"}
	if {$info(type) eq "cache"} {
		::Sqlite::cache set "cache" $info(db) $info(csize)
	} else {
		::Sqlite::cache set "default cache" $info(db) $info(dsize)
	}
}

;#>>>


;#>>>


namespace eval Cmd {};#<<<

# Create & Open New DB
proc Cmd::createDB {version} {
	set file [Sqlite::create $version]
	if {$file ne {}} {
		RecentFile::append $file
		Cmd::updateFrames
		Table::clear
	}
	return 0
}

# 
# Open DB
proc Cmd::openDB {filename {version {}}} {
	if {$filename eq {}} {
		set filename [tk_getOpenFile -filetypes $::pref(openTypeSqlite)]
		if { $filename eq {} } return ;# Cancel
	}
	if {[catch {::Sqlite::tryOpen $filename $version} ret]} {
		::Message::show -type apperror \
		-message "open error" \
		-detail $ret
		return
	}
	set version $ret
	::Sqlite::close
	::Cmd::clearDatabaseInfo

	::Cmd::selectEncoding $::database(encoding)
	::Sqlite::open $filename $version
	::Cmd::updateFrames
	::RecentFile::append $filename
	return 0
}

# Close DB
proc Cmd::closeDB {} {
	::Sqlite::close
	::Cmd::clearDatabaseInfo
	::Cmd::updateFrames
	return 0
}

# attach DB
proc Cmd::attachDB {} {
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	set version [Sqlite::getCurrentVersion]
	set file [tk_getOpenFile -filetypes $::pref(openTypeSqlite)]
	if { $file eq {} } return ;# Cancel
	
	# Error handling
	set fileversion [Sqlite::getFileVersion $file]
	if {$fileversion eq {}} {
		::Message::show -type apperror -detail $file \
		-message [msgcat::mc "The file is not a SQLite file."]
		return
	}
	if {$fileversion ne $version} {
		::Message::show -type apperror -detail $file \
		-message [msgcat::mc "Failed to attach the db. The db version is not same as current db."]
		return
	}
	
	# Get attached database name
	set name [GetString::show "DB Name" [msgcat::mc "Enter database name."]]
	if {$name eq ""} {return} ;#canceled

	# run SQL
	# FIXME:if $name has some words. use first one only.
	# Because if db name has white space, we can't detach it on sqlite2. 
	# See Cmd::detachDB proc too.
	set query "attach '$file' as [lindex $name 0];"
	if {[Sqlite::evalQuery $query] != 0} {
		return
	}
	updateFrames
}

# detach DB
proc Cmd::detachDB {} {
	set dblist [Dbtree::getSelectedDBName]
	
	if {[llength $dblist] == 0} {
		Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please select a database."]
		return
	}
	
	set result [Message::show -buttons {ok cancel} -icon question -textbox 1 \
	-default cancel -message [msgcat::mc "May I detach the database?"]\
	-detail [join $dblist \n]]
	
	if {$result ne "ok"} return

	foreach db [Dbtree::getSelectedDBName] {
		# FIXME:If we quoted db name, we got error in sqlite2.
		# Maybe This is a sqlite2 bug. If db name has white space,
		# We can't detach the db except for re-open a main db.
		# See Cmd::attachDB proc too.
		Sqlite::evalQuery "detach $db"
	}
	updateFrames
	Table::clear
}

# Clean database file info
proc Cmd::clearDatabaseInfo {} {
	array set ::database {
		path  {} name {}
	}
	# don't clean encoding and version
	::Table::clear
	return 0
}

# Dump SQL
proc Cmd::exportSQL {} {
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	set file [tk_getSaveFile -filetypes $::pref(openTypeSql)]
	if {$file eq {}} return

	set fp [open $file w]
	fconfigure $fp -encoding $::database(encoding)
	if {$::database(encoding) eq "unicode"} {
		::Util::Unicode::writeUTF16BOM $fp
	}
	catch {Sqlite::export main $fp}
	close $fp
}

proc Cmd::importSQL {} {
	if {$::database(name) eq {}} {
		::Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please open a Database file."]
		return
	}
	set file [tk_getOpenFile -filetypes $::pref(openTypeSql)]
	if {$file eq {}} return
	
	set fp [open $file r]
	fconfigure $fp -encoding $::database(encoding)
	set isLittleEndian [::Util::Unicode::isLittleEndian $fp]
	if {$::database(encoding) eq "unicode"} {
		set query [::Util::Unicode::readAll $fp $isLittleEndian]
	} else {
		set query [read $fp]
	}
	close $fp
	Sqlite::import $query
	updateFrames
}

# Update property table. wrapper of Property::update
proc Cmd::updateProperty {} {
	dputs {updateProperty:[lindex [Dbtree::getSelection] 0]}
	return [Property::update [lindex [Dbtree::getSelection] 0]]
}

# Update All frames
proc Cmd::updateFrames {} {
	if {$::database(name) eq {}} {
		wm title . $::pref(appname)
	} else {
		wm title . "$::pref(appname) : $::database(path)" 
		# check sqlite internal encoding
		if {$::database(version) == 2} {
			if {$::database(encoding) eq "unicode"} {
				::Cmd::selectEncoding $::SYSENCODING
			}
		}
		if {$::database(version) == 3} {
			::Sqlite::evalQuery {PRAGMA encoding;}
			if {$::Sqlite::data eq "UTF-8"} {
				if {$::database(encoding) eq "unicode"} {
					::Cmd::selectEncoding "utf-8"
				}
			} else {
				if {$::database(encoding) ne "unicode"} {
					::Cmd::selectEncoding "unicode"
				}
			}
		}
	}

	::Dbtree::update
	::Property::update [lindex [::Dbtree::getSelection] 0]
	::Statusbar::update
	return 0
}

proc Cmd::getSqlTime {} {
	return $::Sqlite::status(time)
}

proc Cmd::getTableRowCount {} {
	return [expr {[.table.f.table cget -rows] - 1}]
}

# Do SQL and return result list
proc Cmd::evalSQL {query} {
	if {[Sqlite::evalQuery $query] == 0} {
		return $::Sqlite::data
	}
	return [list]
}

# Do SQL and show result at Table.
proc Cmd::executeSqlCmd {} {
	set q [Editor::getText]
	if {[::Cmd::showTable $q] != 0} {
		updateFrames
		return 1
	}
	# add history list
	if {$q ne {}} {History::append $q}

	updateFrames
	Table::focusThis
	Editor::focusThis
	return 0
}

# Delete Table/View
proc Cmd::dropTable {} {
	if {$::database(name) eq {}} {return}
	set tstr {}
	set vstr {}
	set delList [list]
	foreach {tableData} [Dbtree::getSelection] {
		if {[lindex $tableData 2] eq {}} continue
		set db   [lindex $tableData 2 0];# db name
		set type [lindex $tableData 2 1];# table/view
		set name [lindex $tableData 2 2];# table name
		switch -exact -- $type {
			table {lappend delList $type $db $name; append tstr \t$db.$name\n}
			view  {lappend delList $type $db $name; append vstr \t$db.$name\n}
		}
		
	}
	if {[llength $delList] == 0} {
		Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please select tables or views."]
		return 0
	}
	# Confirm before dorpping table
	set md {}
	if {$tstr ne {}} {append md "[msgcat::mc Table]\n" $tstr}
	if {$vstr ne {}} {append md "[msgcat::mc View]\n"  $vstr}
	append md \n
	set result [Message::show -buttons {ok cancel} \
		-icon question -default cancel -textbox 1 \
		-message [msgcat::mc "May I really drop the tables or views?"] -detail $md]
	if {$result ne "ok"} { return 0 }

	set version [Sqlite::getCurrentVersion]
	
	set drop_opend_table 0
	if {$version == 3} {
		foreach {type db name} $delList {
			Sqlite::evalQuery "drop $type '$db'.'$name'"
			if {$db eq $::Table::edit(db) && $name eq $::Table::edit(table)} {
				set drop_opend_table 1
			}
		}
	}

	if {$version == 2} {
		set unDelList [list]
		foreach {type db name} $delList {
			if {$db eq "main" || $db eq "temp"} {
				Sqlite::evalQuery "drop $type '$name'"
			} else {
				if {[llength [Sqlite::findTable $name]] == 2} {
				Sqlite::evalQuery "drop $type '$name'"
				} else {
				lappend unDelList "$type\t$db.$name"
				}
			}
			if {$db eq $::Table::edit(db) && $name eq $::Table::edit(table)} {
				set drop_opend_table 1
			}
		}
		if [llength $unDelList] {
			Message::show -type apperror \
			-message [msgcat::mc "Failed to drop the tables/views. A table/view with tha same name already exists on other database."] \
			-detail [join $unDelList \n]
		}
	}

	if {$drop_opend_table} {
		::Table::clear
	}
	updateFrames
	return 0
}

# Delete Index
proc Cmd::dropIndex {} {
	if {$::database(name) eq {}} {return}
	set table $::Property::tab(index).table
	set selection [lindex [$table curselection] 0]
	set row [$table index $selection row]
	if {$selection eq {} || $row == 0} {
		Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please select an index."]
		return 0
	}
	dputs {deleteIndexRow=$row}
	set indexName $::Property::index($row,0)
	set dbName    $::Property::index($row,3)
	set md $dbName.$indexName
	set result [Message::show -buttons {ok cancel} \
		-icon question -default cancel -textbox 1 \
		-message [msgcat::mc "May I really drop the index?"] -detail $md]
	if {$result eq "ok"} {
		# Sqlite2 does not support dropping index on attached database.
		# Database name was ignored here.
		Sqlite::evalQuery "drop index '$dbName'.'$indexName'"
		updateFrames
	}
	return 0
}

# Delete Trigger
proc Cmd::dropTrigger {} {
	if {$::database(name) eq {}} {return}
	set table $::Property::tab(trigger).table
	set selection [lindex [$table curselection] 0]
	set row [$table index $selection row]
	if {$selection eq {} || $row == 0} {
		Message::show -type appinfo -textbox 0 \
		-message [msgcat::mc "Please select a trigger."]
		return 0
	}
	dputs {deleteTriggerRow=$row}
	set triggerName $::Property::trigger($row,0)
	set dbName      $::Property::trigger($row,2)
	set md $dbName.$triggerName
	set result [Message::show -buttons {ok cancel} \
		-icon question -default cancel -textbox 1 \
		-message [msgcat::mc "May I really drop the trigger?"] -detail $md]
	if {$result eq "ok"} {
		# Sqlite2 does not support dropping trigger on attached database. 
		# Database name was ignored here.
		Sqlite::evalQuery "drop trigger '$dbName'.'$triggerName'"
		updateFrames
	}
	return 0
}

# Show Table(Editable)
proc Cmd::editTable {db table} {
	if {[Table::edit $db $table]} {return 1}
	Table::setTitle "[msgcat::mc Table] [msgcat::mc Edit] \[  $db.$table  \]" $::img(writable)
	Statusbar::update
	Table::focusThis
	return 0
}

# Show View (Not Editable)
proc Cmd::showView {db table} {
	if {[Table::show "SELECT * FROM '$db'.'$table'"]} {return 1}
	Table::setTitle "[msgcat::mc View] [msgcat::mc {Not Editable}] \[  $db.$table  \]" $::img(lock)
	Statusbar::update
	Table::focusThis
	return 0
}

# Show Talble(Not Editable)
proc Cmd::showTable {sql} {
	if {[Table::show $sql]} {return 1}
	Table::setTitle [msgcat::mc Result] $::img(lock)
	Statusbar::update
	return 0
}

# Do Vacuum
proc Cmd::vacuum {} {
	set detailmsg [msgcat::mc "If this DB was big, vacuum will take a long time."]

	set result [::Message::show -buttons {ok cancel} -icon question \
	-default cancel -message [msgcat::mc "Run vacuum?"] -textbox 1 \
	-detail $detailmsg]
	if {$result ne "ok"} return
	
	::Sqlite::vacuum
	updateFrames
}

# Show SQL Statement at Edotor
proc Cmd::showSQLStatement {data} {
	dputs {$data}
	Editor::setText $data
}

# double click of History
proc Cmd::doubleClickHistory {} {
	set sel [History::getSelection]
	if {$sel eq {}} return
	return [Editor::setText [History::getSelection]]
}

# change encoding
proc Cmd::selectEncoding {enc} {
	set ::database(encoding) $enc
	::Table::clear
	updateFrames
	return 0
}

# rebuild encoding menu
proc Cmd::rebuildEncodingMenu {m} {
	$m delete 0 end
	set i 0
	set isUnicode [expr {$::database(encoding) eq {unicode}}]
	if {$::database(name) eq ""} {
		set isUnicode 0
	}
	if {$::database(version) == 2} {
		set isUnicode 0
	}

	foreach enc $::pref(enable_encoding) {
		if {$enc eq {identity}} {continue}

		$m add radiobutton -label $enc \
		-variable ::database(encoding) \
		-command "::Cmd::selectEncoding $enc" \
		-columnbreak [expr {($i%20) == 0}] \
		-state [expr {$isUnicode ? 
			($enc eq "unicode" ? "normal":"disabled") : 
			($enc eq "unicode" ? "disabled":"normal")}]

		incr i
	}
}

# strName is var name, not value
# encoding convert to/from db encoding
# this proc is used for sqlite3 only.
# sqlite2 require changing [encoding system] directly.
# Note : tclsqlite encoding convert
# tclsqlite2 : Set : tcl's utf-8 interface       -> [encoding system]
# tclsqlite2 : Get : [encoding system]           -> tcl's utf-8 interface
# tclsqlite3 : Set : tcl's utf-8 interface       -> sqlite3 db enc (utf8/utf16)
# tclsqlite3 : Get : sqlite3 db enc (utf8/utf16) -> tcl's utf-8 interface

proc Cmd::encodingConvert {dir strName {isList 0}} {
	if {$::database(encoding) eq {utf-8}} {return}
	if {$::database(encoding) eq {unicode}} {return}
	if {$::database(version) == 2} {return}
	
	# Here is only for (SQLite3 && multi-byte)
	upvar $strName str
	switch -exact -- [string tolower $dir] {
		to {
		set str [encoding convertto $::database(encoding) $str]
		set str [encoding convertfrom identity $str]
		}
		from {
		if {$isList} {
			# some multi-byte char has "{", "}" in the 2-byte.
			# They are escaped when they are in a element of list.
			# So we must convert it one by one.
			set i 0
			foreach w $str {
				set w [encoding convertto identity $w]
				set w [encoding convertfrom $::database(encoding) $w]
				lset str $i $w
				incr i
			}
		} else {
			set str [encoding convertto identity $str]
			set str [encoding convertfrom $::database(encoding) $str]
		}
		}
	}
	return 0
}

# Change Current Directory
proc Cmd::changeCureentDir {{dir ""}} {
	if {$dir eq ""} {
		set dir [tk_chooseDirectory -initialdir [pwd] -parent . -mustexist 1]
	}
	if {[file exists $dir]} {
		if {[file isdir $dir]} {
			cd $dir
		}
	}
}

# Exit
proc Cmd::exit {} {
	set session ""
	append session [Cmd::dump]
	append session [Sqlite::dump]
	append session [RecentFile::dump]
	append session [SQLText::dump]
	append session [Table::dump]
	append session [History::dump]
	append session [Tool::Searcher::dump]
	Session::write $session
	tcl_exit 0
}
if {$pref(usesession) != 0} {
rename exit tcl_exit
rename Cmd::exit exit
}

proc Cmd::changeTableLookAndFeel {table} {
	set relief ridge
	switch -exact -- [tk windowingsystem] {
		win32   { set border {0 1 0 1} }
		default { set border {1 0 1 0} }
	}

	set drawmode compatible
	if {$::tk_version >= 8.5 && $::tcl_platform(platform) eq "unix"} {
		if {[info command ::tk::pkgconfig] ne ""} {
			if {[::tk::pkgconfig get fontsystem] eq "xft"} {
				set drawmode slow
			}
		}
	}

	$table configure -highlightthickness 0 -fg black -bg gray60 -anchor w \
		-state disabled -cursor arrow -relief $relief -anchor nw -borderwidth $border \
		-drawmode $drawmode -cursor {} -font TkTextFont -ipadx 1 \
		-rowseparator \"\n\" -colseparator \"\t\" -selcmd {return \"%s\"}

	set tbg [style lookup . -background]
	set tfg [style lookup . -foreground]

	$table tag col default 0
	$table tag config default -relief $relief -fg black -bg white
	$table tag config title   -relief raised  -fg $tfg  -bg $tbg -borderwidth 1
	$table tag config sel     -relief $relief -fg black -bg SkyBlue
	$table tag config newcol  -relief $relief -fg black -bg SkyBlue
	$table tag config active  -relief $relief -fg black -bg SkyBlue
	$table tag config pkey    -image $::img(key) -anchor center
	$table tag lower default
	$table tag raise pkey
	eval $table tag col default [Util::range 1 200]

	bind $table <MouseWheel> {
		%W yview scroll [expr {-(%D/120) * 4}] units
	}
}

# Create treectrl
proc Cmd::createTreectrl {parentFrameName {scrollbartype y}} {
	if {[info command $parentFrameName] eq ""} {
		set f [ttk::frame $parentFrameName]
	} else {
		set f $parentFrameName
	}
	set tlist [::treectrl $f.tlist -relief flat -width 100 -height 100 \
	-background white -highlightthickness 0 -showroot no \
	 -showbuttons no -showlines no -itemwidthequal no \
	 -showheader yes -takefocus 1 -xscrollincrement 20 \
	-font TkTextFont]

	switch -exact -- $scrollbartype {
		x {
			set x [ttk::scrollbar $f.x -takefocus 0 -orient horizontal -command "$tlist xview"]
			$tlist configure -xscrollcommand "$x set"
			pack $tlist -side top -fill both -expand yes
			pack $x     -side top -fill x
		}
		y {
			set y [ttk::scrollbar $f.y -takefocus 0 -orient vertical -command "$tlist yview"]
			$tlist configure -yscrollcommand "$y set"
			pack $tlist -side left -fill both -expand yes
			pack $y     -side left -fill y
		}
		both {
			set bg [ttk::frame $f.bg]
			set x [ttk::scrollbar $f.x -takefocus 0 -orient horizontal -command "$tlist xview"]
			set y [ttk::scrollbar $f.y -takefocus 0 -orient vertical   -command "$tlist yview"]
			$tlist configure -xscrollcommand "$x set" \
							 -yscrollcommand "$y set"
			grid $tlist -row 0 -column 0 -sticky news
			grid $y     -row 0 -column 1 -sticky ns
			grid $x     -row 1 -column 0 -sticky ew
			grid $bg    -row 1 -column 1 -sticky news
			grid rowconfigure    $f 0 -weight 1
			grid columnconfigure $f 0 -weight 1
			grid propagate $f 0
		}
	}

	switch -exact -- $::tile_currentTheme {
		xpnative  {
			$tlist configure -border 0 -usetheme 1;
			$f configure -style TEntry -border 1 -padding 1
		}
		winnative { $tlist configure -border 0; $f configure -relief sunken -border 2 }
		clam      { $tlist configure -border 0; $f configure -style TEntry -border 2 }
		classic   { $tlist configure -relief sunken -border 2; $f configure -relief flat -border 0 }
		aqua      {
			$f configure -style TEntry -border 0
			$tlist configure -border 0 -usetheme 1
		}
		default   {
			$f configure -style TEntry -border 0
			$tlist configure -border 1 -usetheme 0 -relief flat
		}
	}
	Cmd::bindTextFocusTtkEntry $tlist $f
	return $f
}

# This proc will be called after columns are created.
proc Cmd::changeTreectrlHeaderLoolAndFeele {tree} {
	set outlinecolor [::Util::Color::median $::pref(tile_bg) $::pref(tile_bg) #FFFFFF]
	switch -exact -- $::tile_currentTheme {
		xpnative  {set border 2; set textpady 2}
		winnative {set border 2; set textpady 0}
		clam      {set border 1; set textpady 2}
		classic   {set border 2; set textpady 3}
		aqua      {set border 2; set textpady 0; set outlinecolor "#E5E5E5"}
		default   {set border 1; set textpady 3}
	}

	$tree column configure all -font TkHeadingFont \
		-border $border \
		-textpady $textpady \
		-background [list \
			[style lookup TButton -background active] active \
			[style lookup TButton -background pressed] pressed \
			[style lookup TButton -background ""] {}]

	# create basic elements and styles
	$tree element create eIcon image
	$tree element create eTxt text -fill [list $::pref(tile_selectfg) selected] \
		-lines 1 
	$tree element create eSel rect -fill [list $::pref(tile_selectbg) selected] \
		-height 17 -open nw -outline $outlinecolor -outlinewidth 1
	$tree element create eWin window -destroy 1

	set S [$tree style create STYLE_ICONTEXT]
	$tree style elements $S {eSel eIcon eTxt}
	$tree style layout $S eIcon -expand ns -padx {1 0}
	$tree style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$tree style layout $S eSel -union [list eIcon eTxt] \
		-iexpand ns -ipadx 2 -ipady 1 -padx {1 4}

	set S [$tree style create STYLE_TEXT]
	$tree style elements $S {eSel eTxt}
	$tree style layout $S eTxt -squeeze x -expand ns -padx {2 0}
	$tree style layout $S eSel -detach yes -iexpand xy

	set S [$tree style create STYLE_WIN]
	$tree style elements $S {eSel eWin}
	$tree style layout $S eWin -expand ns -squeeze x -padx {2 0}
	$tree style layout $S eSel -union {eWin} -iexpand news -detach yes
}

# Change selection syle of element of treectrl.
# FIXME:treectrl2.2.3 still has a seg fault bug in <Selection>
# bind script with tk8.5. Don't use %S. 
proc Cmd::changeSelectionStyleInTreectrl {tree column element type selectStyle unselectStyle} {
	foreach item [$tree item children root] {
		if {[$tree selection includes $item]} {
			set check [$tree item element cget $item $column $element $type]
			$check configure -style $selectStyle
		} else {
			set check [$tree item element cget $item $column $element $type]
			$check configure -style $unselectStyle
		}
	}
}

proc Cmd::bindTextFocusTtkEntry {text frame} {
	bind $text <FocusIn>  [list + $frame state focus]
	bind $text <FocusOut> [list + $frame state !focus]
	return
}

# dump main window infomation
proc Cmd::dump {} {
	set session ""
	if {[tk windowingsystem] eq "x11"} {
		append session "wm withdraw .\n"
		append session "wm geometry . [wm geometry .]\n"
		append session "wm deiconify .\n"
		append session "update\n"
		append session ".root  sashpos 0 [.root  sashpos 0]\n"
		append session ".left  sashpos 0 [.left  sashpos 0]\n"
		append session ".right sashpos 0 [.right sashpos 0]\n"
		append session ".right sashpos 1 [.right sashpos 1]\n"
	} else {
		append session "wm geometry . [winfo width .]x[winfo height .]+10000+10000\n"
		append session "wm deiconify .\n"
		append session "update\n"
		append session ".root  sashpos 0 [.root  sashpos 0]\n"
		append session ".left  sashpos 0 [.left  sashpos 0]\n"
		append session ".right sashpos 0 [.right sashpos 0]\n"
		append session ".right sashpos 1 [.right sashpos 1]\n"
		append session "wm geometry . +[winfo x .]+[winfo y .]\n"
		append session "update\n"
	}
	append session "set ::pref(enable_encoding) [list $::pref(enable_encoding)]\n"
	if {$::database(name) ne {}} {
	append session "set ::database(encoding) $::database(encoding)\n"
	append session "set ::pref(open_file) {{$::database(path)} $::database(version)}\n"
	if {[file dirname $::database(path)] ne [pwd]} {
	append session "set ::pref(init_dir) {[pwd]}\n"
	}
	}
	return $session
}
;#>>>


;#<<<
proc main {} {
	# create menu
	# -------------------------------------------------
	menu .mbar -border 0
	.mbar add cascade -label [msgcat::mc File] -menu .mbar.file
;#    .mbar add cascade -label [msgcat::mc Edit] -menu .mbar.edit
	.mbar add cascade -label [msgcat::mc Database] -menu .mbar.db
;#    .mbar add cascade -label [msgcat::mc Table] -menu .mbar.table
	.mbar add cascade -label [msgcat::mc Tool] -menu .mbar.tool
	.mbar add cascade -label [msgcat::mc Help] -menu .mbar.help
	
	# file
	menu .mbar.file
	.mbar.file add command -label [msgcat::mc "New SQLite2 DB"]... -command {Cmd::createDB 2}
	.mbar.file add command -label [msgcat::mc "New SQLite3 DB"]... -command {Cmd::createDB 3}
	.mbar.file add command -label [msgcat::mc Open]... -command {Cmd::openDB {}}
	.mbar.file add cascade -label [msgcat::mc "Recent Files"] -menu [::RecentFile::RecentFile [list]]
	.mbar.file add separator
	.mbar.file add command -label [msgcat::mc "Change Current Dir"]... -command {Cmd::changeCureentDir}
	.mbar.file add separator
	.mbar.file add command -label [msgcat::mc Attach]... -command {Cmd::attachDB}
	.mbar.file add command -label [msgcat::mc Detach]... -command {Cmd::detachDB}
	.mbar.file add separator
	.mbar.file add cascade -label [msgcat::mc Import] -menu .mbar.file.import
	.mbar.file add cascade -label [msgcat::mc Export] -menu .mbar.file.export
	.mbar.file add separator
	.mbar.file add command -label [msgcat::mc Close] -command {Cmd::closeDB}
	.mbar.file add separator
	.mbar.file add command -label [msgcat::mc Exit] -accelerator "$::pref(shortmod)+Q" -command {exit}

	menu .mbar.file.import
    menu .mbar.file.export
	.mbar.file.import add command -label [msgcat::mc SQL]... -command {Cmd::importSQL}
	.mbar.file.export add command -label [msgcat::mc SQL]... -command {Cmd::exportSQL}
	.mbar.file.import add command -label [msgcat::mc "Text File"]... -command {GUICmd::ImportText::run}
	.mbar.file.export add command -label [msgcat::mc "Text File"]... -command {GUICmd::ExportText::run}

	bind . <$::pref(modifier)-q> {exit}

	if {$::Sqlite::status(haveSqlite2) == 0} {
		.mbar.file entryconfigure 0 -state disabled
	}
	if {$::Sqlite::status(haveSqlite3) == 0} {
		.mbar.file entryconfigure 1 -state disabled
	}

	# edit
	#menu .mbar.edit
	
	# database
	menu .mbar.db
	#TODO
	.mbar.db add command -label [msgcat::mc "Run SQL"] -command {Cmd::executeSqlCmd}
	.mbar.db add separator
	.mbar.db add command -label [msgcat::mc "Create Table"]... -command {::GUICmd::TableBuilder::run create}
;#    .mbar.db add command -label [msgcat::mc "Create View"] -state disabled
	.mbar.db add command -label [msgcat::mc "Create Index"]... -command {::GUICmd::CreateIndex::run}
	.mbar.db add command -label [msgcat::mc "Create Trigger"]... -command {::GUICmd::CreateTrigger::run}
	.mbar.db add separator
	.mbar.db add command -label [msgcat::mc "Modify Table Schema"]... -command {::GUICmd::TableBuilder::run modify}
	.mbar.db add separator
	.mbar.db add command -label [msgcat::mc "Drop Table/View"]... -command {Cmd::dropTable}
	.mbar.db add command -label [msgcat::mc "Drop Index"]... -command {Cmd::dropIndex}
	.mbar.db add command -label [msgcat::mc "Drop Trigger"]... -command {::Cmd::dropTrigger}
	.mbar.db add separator
	.mbar.db add command -label [msgcat::mc "Set Cache Size"]... -command {::GUICmd::SetCacheSize::run}
	.mbar.db add separator
	.mbar.db add command -label [msgcat::mc VACUUM] -command {::Cmd::vacuum}
	.mbar.db add command -label [msgcat::mc "Check Integrity"] -command {::Sqlite::checkIntegrity}
	.mbar.db add separator
	.mbar.db add cascade -label [msgcat::mc Encoding] -menu .mbar.db.encoding

	# database/encoding 
	set _me .mbar.db.encoding
	menu $_me -postcommand "Cmd::rebuildEncodingMenu $_me"
	
	# table
	menu .mbar.table

	# tool
	menu .mbar.tool
	.mbar.tool add command -label [msgcat::mc Preference]... -command GUICmd::Preference::run
	.mbar.tool add separator
	.mbar.tool add command -label [msgcat::mc Search]... -accelerator {F3} \
		-command {GUICmd::Search}
	bind . <F3> {GUICmd::Search}

	.mbar.tool add command -label [msgcat::mc "Data Editor"]... -accelerator {F4} \
		-command {Tool::DataEditor::run}
	bind . <F4> {Tool::DataEditor::run}

	# Only Debug mode
	deval {
	.mbar.tool add separator
	if {[info command console] ne ""} {
		.mbar.tool add command -label Console -command {console show}
	} else {
		uplevel 1 {catch {source [file join \
				[file dir [info nameofexe]] tkcon.tcl]}}
		if {[info command tkcon] ne ""} {
		.mbar.tool add command -label Console -command {tkcon show}
		}
	}
	.mbar.tool add command -label encoding -command {
	tk_messageBox -message $::database(encoding)\nsystem:[encoding system]}
	}

	#help
	menu .mbar.help
	.mbar.help add command -label {About...} -command {
		set m [list]
		lappend m "Tcl $tcl_patchLevel"
		lappend m "Tk $tk_patchLevel"
		foreach {n v} [::Sqlite::getLibraryVersion] {
		if {$v eq {}} {
			set v " not found"
		}
		lappend m "tclsqlite$n $v"
		}
		if {![catch {package present Tktable} v]} {
		lappend m "Tktable $v"
		}
		if {![catch {package present tile} v]} {
		lappend m "Tile $v"
		}
		if {![catch {package present treectrl} v]} {
		lappend m "Treectrl $v"
		}
		if {![catch {package present dict} v]} {
		lappend m "Dict $v"
		}
		if {![catch {package present Trf} v]} {
		lappend m "Trf $v"
		}
		if {![catch {package present Img} v]} {
		lappend m "Img $v"
		}
		set h [expr {[llength $m] / 2}]
			if {([llength $m] % 2) != 0} {
			incr h
		}
		set v ""
		for {set i 0} {$i < $h} {incr i} {
			append v \
			[format %-30s [lindex $m $i]] "\t"\
			[lindex $m [expr {$i + $h}]] "\n"
		}

		Message::show -buttons ok -message "TkSQLite $::VERSION" \
		-detail $v$COPYRIGHT -default ok -textbox 0 \
		-title "About TkSQLite"

		unset -nocomplain -- m n v i h
	}

	. configure -menu .mbar
	wm protocol . WM_DELETE_WINDOW { exit }
	
	# create status bar
	# -------------------------------------------------
	pack [Statusbar::Statusbar] -side bottom -fill x -expand no
	
	# create widgets
	# -------------------------------------------------
	if {$::pref(tile_version) >= 0.8} {
	ttk::panedwindow .root -orient horizontal
	.root add [ttk::panedwindow .left  -orient vertical] -weight 0
	.root add [ttk::panedwindow .right -orient vertical] -weight 1
	} else {
	ttk::paned .root -orient horizontal
	.root add [ttk::paned .left  -orient vertical] -weight 0
	.root add [ttk::paned .right -orient vertical] -weight 1
	}
	.left add [Dbtree::Dbtree]     -weight 0
	.left add [Property::Property] -weight 1
	.right add [Editor::Editor]   -weight 0
	.right add [Table::Table]     -weight 1
	.right add [History::History] -weight 0
	pack .root -expand yes -fill both

	bind . <Control-bracketleft> "event generate %W <Key-Escape>; break"
}

# Start Here
wm withdraw .
wm title . $pref(appname)
main
set _currentdir [pwd]
if {$pref(usesession) != 0 && [file exists $::Session::file]} {
	if {[catch {Session::read} m]} {
		Message::show -type apperror -message "initialize error" -detail $m
	}

	# old pref hacks
	set pref(enable_encoding) [string map {"ucs-2be" "" "X11ControlChars" ""} $pref(enable_encoding)]
	if {[array exists ::Editor(info)]} {
		foreach {key value} [array get ::Editor(info)] {
			set ::SQLText::info($key) $value
		}
		# array unset ::Editor::info
	}
} else {
	wm withdraw .
	wm geometry . 640x480+10+10
	wm deiconify .
	update
	.root sashpos 0 200
	.left sashpos 0 200
	.right sashpos 0 70
	.right sashpos 1 390
}

if {[llength $argv] > 0} {
	if {$database(name) ne ""} { Cmd::closeDB }
	set _file [lindex $argv 0]
	if {[namespace exists ::starkit]} {
		set _file [encoding convertfrom $::SYSENCODING $_file]
		Cmd::openDB [file normalize [file join $_startdir $_file]]
	} else {
		Cmd::openDB [file normalize [file join $_currentdir $_file]]
	}
	unset _file
} else {
	set _file [lindex $pref(open_file) 0]
	set _ver  [lindex $pref(open_file) 1]
	if {$_file ne ""} { Cmd::openDB $_file $_ver }
	if {$pref(init_dir) ne ""} {
		Cmd::changeCureentDir $pref(init_dir)
	}
	unset _file
	unset _ver
}

wm deiconify .
focus -force .

;#>>>
