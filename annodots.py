from PIL import Image, ImageDraw
from xml.etree import ElementTree
import os


def save_dots(anno, imagedir, savedir):
    red = (180, 0, 0, 128)
    green = (0, 180, 50)
    white = (200, 200, 200)
    black = (30, 30, 30)
    centroid_radius = 3
    point_radius = 2

    imagename = next(anno.find('image').find('name').itertext())
    im = Image.open(os.path.join(imagedir, imagename))
    draw = ImageDraw.Draw(im)

    annorect = anno.find('annorect')
    #draw rectangle
    x1 = int(annorect.find('x1').text)
    y1 = int(annorect.find('y1').text)
    x2 = int(annorect.find('x2').text)
    y2 = int(annorect.find('y2').text)
    draw.rectangle([(x1, y1), (x2, y2)], outline=red)

    #draw center of mass
    objpos = annorect.find('objpos')
    objx = int(objpos.find('x').text)
    objy = int(objpos.find('y').text)
    draw.ellipse([(objx - centroid_radius, objy - centroid_radius),
                    (objx + centroid_radius, objy + centroid_radius)], fill=red)

    #draw points and their labels
    annopoints = annorect.find('annopoints')
    for point in annopoints:
        id_ = point.find('id').text
        x = int(point.find('x').text)
        y = int(point.find('y').text)
        draw.ellipse([(x - point_radius, y - point_radius),
                        (x + point_radius, y + point_radius)], fill=green)
        draw_text_outline(draw, x + point_radius, y - point_radius - 10,
                            id_, white, black)

    imagename_dots = imagename[:-4] + '_dots' + imagename[-4:]
    im.save(os.path.join(savedir, imagename_dots))


def draw_text_outline(draw, x, y, text, fill, outline):
    draw.text((x - 1, y), text, fill=outline)
    draw.text((x + 1, y), text, fill=outline)
    draw.text((x, y - 1), text, fill=outline)
    draw.text((x, y + 1), text, fill=outline)
    draw.text((x, y), text, fill=fill)


if __name__ == '__main__':
    import sys

    annolist = ElementTree.parse(sys.argv[1]).getroot()
    imagedir = sys.argv[2]
    savedir = sys.argv[3]
    for anno in annolist:
        save_dots(anno, imagedir, savedir)
