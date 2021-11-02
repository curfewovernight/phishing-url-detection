# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class frameMain
###########################################################################

class frameMain ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Phishing URL Test", pos = wx.DefaultPosition, size = wx.Size( 586,388 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		bSizerFrameMain = wx.BoxSizer( wx.VERTICAL )

		bSizerMainFrame = wx.BoxSizer( wx.VERTICAL )

		self.m_panelMain = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,300 ), wx.TAB_TRAVERSAL )
		bSizerMainPanel = wx.BoxSizer( wx.VERTICAL )

		bSizerPanelMain = wx.BoxSizer( wx.HORIZONTAL )

		bSizer6 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText2 = wx.StaticText( self.m_panelMain, wx.ID_ANY, u"Enter URL :", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )

		bSizer6.Add( self.m_staticText2, 0, wx.ALL, 9 )


		bSizerPanelMain.Add( bSizer6, 0, wx.ALL, 0 )

		bSizer611 = wx.BoxSizer( wx.VERTICAL )

		self.m_textCtrl21 = wx.TextCtrl( self.m_panelMain, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer611.Add( self.m_textCtrl21, 0, wx.ALL|wx.EXPAND, 5 )


		bSizerPanelMain.Add( bSizer611, 1, 0, 0 )


		bSizerMainPanel.Add( bSizerPanelMain, 0, wx.EXPAND|wx.TOP, 25 )

		bSizerPanelMain1 = wx.BoxSizer( wx.VERTICAL )

		bSizer62 = wx.BoxSizer( wx.VERTICAL )

		self.m_button12 = wx.Button( self.m_panelMain, wx.ID_ANY, u"Predict", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer62.Add( self.m_button12, 0, wx.ALL|wx.EXPAND, 5 )


		bSizerPanelMain1.Add( bSizer62, 0, wx.ALL|wx.EXPAND, 0 )


		bSizerMainPanel.Add( bSizerPanelMain1, 0, wx.EXPAND, 5 )

		bSizerPanelMain11 = wx.BoxSizer( wx.VERTICAL )

		bSizer621 = wx.BoxSizer( wx.HORIZONTAL )

		self.fstClass = wx.StaticText( self.m_panelMain, wx.ID_ANY, u"MyLabel", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.fstClass.Wrap( -1 )

		bSizer621.Add( self.fstClass, 1, wx.ALL, 9 )

		self._1stResult = wx.TextCtrl( self.m_panelMain, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 420,-1 ), wx.TE_PROCESS_TAB|wx.TE_READONLY|wx.TE_RICH )
		bSizer621.Add( self._1stResult, 0, wx.ALL, 5 )


		bSizerPanelMain11.Add( bSizer621, 0, wx.ALL|wx.EXPAND, 0 )

		bSizer6212 = wx.BoxSizer( wx.HORIZONTAL )

		self.SecClass = wx.StaticText( self.m_panelMain, wx.ID_ANY, u"MyLabel", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.SecClass.Wrap( -1 )

		bSizer6212.Add( self.SecClass, 1, wx.ALL, 9 )

		self._2ndResult = wx.TextCtrl( self.m_panelMain, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 420,-1 ), wx.TE_PROCESS_TAB|wx.TE_READONLY|wx.TE_RICH )
		bSizer6212.Add( self._2ndResult, 0, wx.ALL, 5 )


		bSizerPanelMain11.Add( bSizer6212, 0, wx.EXPAND, 5 )

		bSizer62121 = wx.BoxSizer( wx.HORIZONTAL )

		self.ThrdClass = wx.StaticText( self.m_panelMain, wx.ID_ANY, u"MyLabel", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.ThrdClass.Wrap( -1 )

		bSizer62121.Add( self.ThrdClass, 1, wx.ALL, 9 )

		self._3rdResult = wx.TextCtrl( self.m_panelMain, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 420,-1 ), wx.TE_PROCESS_TAB|wx.TE_READONLY|wx.TE_RICH )
		bSizer62121.Add( self._3rdResult, 0, wx.ALL, 5 )


		bSizerPanelMain11.Add( bSizer62121, 0, wx.EXPAND, 5 )

		bSizer6211 = wx.BoxSizer( wx.HORIZONTAL )

		self.FthClass = wx.StaticText( self.m_panelMain, wx.ID_ANY, u"MyLabel", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.FthClass.Wrap( -1 )

		bSizer6211.Add( self.FthClass, 1, wx.ALL, 9 )

		self._4thResult = wx.TextCtrl( self.m_panelMain, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 420,-1 ), wx.TE_PROCESS_TAB|wx.TE_READONLY|wx.TE_RICH )
		bSizer6211.Add( self._4thResult, 0, wx.ALL, 5 )


		bSizerPanelMain11.Add( bSizer6211, 0, wx.EXPAND, 5 )


		bSizerMainPanel.Add( bSizerPanelMain11, 1, wx.EXPAND|wx.TOP, 40 )


		self.m_panelMain.SetSizer( bSizerMainPanel )
		self.m_panelMain.Layout()
		bSizerMainFrame.Add( self.m_panelMain, 1, wx.EXPAND |wx.ALL, 0 )


		bSizerFrameMain.Add( bSizerMainFrame, 1, wx.ALL|wx.EXPAND, 0 )


		self.SetSizer( bSizerFrameMain )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.m_button12.Bind( wx.EVT_BUTTON, self.OnPredict )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def OnPredict( self, event ):
		event.Skip()


