# coding=UTF-8
import matplotlib.pyplot as plot

class ImagerClass(object):

    #Save the image data on the disk.
    def WriteImage(ImageData, filename):
        if ImageData.ndim==1:
            plot.bar(range(256), ImageData)
            plot.savefig(filename)
        else:
            plot.gray()
            plot.axis("off")
            plot.imshow(ImageData)
            plot.savefig(filename)

        plot.close()

        return True
