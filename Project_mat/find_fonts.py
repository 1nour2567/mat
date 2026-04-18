import matplotlib.font_manager

# 查找所有可用的字体
fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 打印中文字体
print("系统中可用的中文字体：")
for font in fonts:
    try:
        font_name = matplotlib.font_manager.FontProperties(fname=font).get_name()
        if any(char >= '\u4e00' and char <= '\u9fff' for char in font_name):
            print(f"字体路径: {font}")
            print(f"字体名称: {font_name}")
            print()
    except:
        pass

# 打印所有字体
print("\n系统中所有可用的字体：")
font_names = set()
for font in fonts:
    try:
        font_name = matplotlib.font_manager.FontProperties(fname=font).get_name()
        font_names.add(font_name)
    except:
        pass

for name in sorted(font_names):
    print(name)
