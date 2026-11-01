from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
RUN=Path(__file__).resolve().parent
import sys
folder=sys.argv[1] if len(sys.argv)>1 else 'render'
files=sorted((RUN/folder).glob('p-*.png'))
for start in range(0,len(files),12):
    sheet=Image.new('RGB',(1000,1120),'#e5e8eb');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(files[start:start+12]):
        im=Image.open(p).convert('RGB');im.thumbnail((240,335))
        x=(j%4)*250+(250-im.width)//2;y=(j//4)*370+25
        sheet.paste(im,(x,y));draw.text(((j%4)*250+10,(j//4)*370+7),p.stem,fill='black')
    sheet.save(RUN/f'{folder}-contacto-{start//12+1}.jpg')
print(len(files))
