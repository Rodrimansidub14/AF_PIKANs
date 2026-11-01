from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
root=Path(__file__).resolve().parent/'render'
pages=sorted(root.glob('page-*.png'))
for offset in range(0,len(pages),6):
    sheet=Image.new('RGB',(1320,1740),'#e7e7e7')
    pen=ImageDraw.Draw(sheet)
    for j,p in enumerate(pages[offset:offset+6]):
        im=Image.open(p).convert('RGB'); im.thumbnail((640,825))
        x=20+(j%2)*660; y=30+(j//2)*575
        # Preserve page aspect while fitting three rows.
        im.thumbnail((630,535))
        sheet.paste(im,(x+(630-im.width)//2,y))
        pen.text((x+10,y-18),p.stem,fill='black')
    sheet.save(root/f'contact-{offset//6+1}.png')
print('Contact sheets:',(len(pages)+5)//6)
