import wx


class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size = (500,300))
        self.panel = MyPanel(self)


class MyPanel(wx.Panel):
    def __init__(self, parent):
        super(MyPanel, self).__init__(parent)

        self.label = wx.StaticText(self, label = "jyagefubusfe")


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(parent=None, title="sjdfy")
        self.frame.Show()
        return True

app = MyApp()
app.MainLoop()