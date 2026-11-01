from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
root=Path('revision-visual');files=sorted(root.glob('pagina-*.png'))
for g in range(0,len(files),12):
    canvas=Image.new('RGB',(960,1740),'#dddddd');draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(files[g:g+12]):
        im=Image.open(p).convert('RGB');im.thumbnail((300,405));x=(j%3)*320+10;y=(j//3)*435+25
        canvas.paste(im,(x,y));draw.text((x,y-18),p.stem,fill='black')
    canvas.save(root/f'contacto-{g//12+1}.png')
print(len(files),'pages rendered')
