import sys
import cv2
import numpy as np
# try: #delete this package after testing
#     from basler_cam import VideoStream
# except:
#     print("Basler package not found")
#     sys.exit(0)

class Image:
    """docstring forImage."""
    def __init__(self):
        """Default kernel size is (5,5) """
        self._img=''
        self._mask=""
        self.kernel_size=(5,5)
        self.threshold_method=[cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
        self.contour_methods=[cv2.RETR_CCOMP,cv2.RETR_FLOODFILL,cv2.RETR_TREE,cv2.RETR_EXTERNAL,cv2.RETR_LIST]
        self.contour_filter_methods=[cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_NONE]

    def set_kernel_size(self,*ksize):
        """
        set kernel size
        set_kernel_size(self,*ksize)
        use odd numbers
        """
        self.kernel_size=(ksize[0],ksize[1])

    def set_image(self,img):
        """
        set image
        set_image(image)
        """
        self._img=img

    def shape(self):
        """
        Returns:
            shape of image
        """
        try:
            return (self._img.shape)

        except Exception as e:
            print("set image first using : set_image(img)")
            print(e)
            sys.exit(0)

    def segmentation(self,*hsv):
        """
        How to use segmentation():
            segmentation(low_threshold_value,high_threshold_value,segmentation_method)

        Note: Segmentation parameter(method) is integer from 0 to 4

        Segmentation methods for gray scale image:
            0 = cv2.THRESH_BINARY
            1 = cv2.THRESH_BINARY_INV
            2 = cv2.THRESH_TRUNC
            3 = cv2.THRESH_TOZERO
            4 = cv2.THRESH_TOZERO_INV

        Returns:
            mask image
        """
        try:
            print("len hsv",len(hsv),hsv)
            if len(hsv)==3 or len(hsv)==6:
                channel_size=len(self.shape())
                if channel_size==3:
                    print(self.kernel_size)
                    blur_image = cv2.GaussianBlur(self._img, (self.kernel_size[0], self.kernel_size[1]), 0)
                    hsv_img = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_img, (hsv[0],hsv [1],hsv[2]),(hsv[3],hsv[4],hsv[5]))

                else:
                    mask = cv2.inRange(self._img,hsv[0],hsv[1])

                self._mask=mask
                return mask
            else:
                print("Segmentation() accepts only 3 or 6 parameter")
        except Exception as e:
            print("set image first using : set_image(img)")
            print(e)
            sys.exit(0)

    def area(self,method=2,filter=0):
        """
        How to use area():
            area(method,filter)

        Note: area parameter(method) is integer from 0 to 4
              area parameter(filter) is integer from 0 to 2

        area methods needs two parameter method and filter :
        default parameter is method = cv2.RETR_TREE and filter = cv2.CHAIN_APPROX_SIMPLE
        methods:
            0 = cv2.RETR_CCOMP
            1 = cv2.RETR_FLOODFILL
            2 = cv2.RETR_TREE
            3 = cv2.RETR_EXTERNAL
            4 = cv2.RETR_LIST
        filter:
            0 = cv2.CHAIN_APPROX_SIMPLE
            1 = cv2.CHAIN_APPROX_NONE

        Returns:
            area of big contour
        """
        try:
            _, cnts, _ = cv2.findContours(self._mask, self.contour_methods[method], self.contour_filter_methods[filter])
            try:
                max_cnts = max(cnts, key=cv2.contourArea)
                area=cv2.contourArea(max_cnts)
            except:
                area=0
            return area
        except:
            print("Error : Do segmentation() first and then use area()")
            sys.exit(0)

    def find_contour(self,method=2,filter=0):
        """
        How to use find_contour():
            find_contour(method,filter)

        Note: find_contour parameter(method) is integer from 0 to 4
              find_contour parameter(filter) is integer from 0 to 2

        find_contour() methods needs two parameter method and filter :
        default parameter is method = cv2.RETR_TREE and filter = cv2.CHAIN_APPROX_SIMPLE

        methods:
            0 = cv2.RETR_CCOMP
            1 = cv2.RETR_FLOODFILL
            2 = cv2.RETR_TREE
            3 = cv2.RETR_EXTERNAL
            4 = cv2.RETR_LIST
        filter:
            0 = cv2.CHAIN_APPROX_SIMPLE
            1 = cv2.CHAIN_APPROX_NONE

        Returns:
            list of all contours
        """
        try:
            _, cnts, _ = cv2.findContours(self._mask, self.contour_methods[method], self.contour_filter_methods[filter])
            return cnts
        except:
            print("Error : Do segmentation() first and then use find_contour()")
            sys.exit(0)

    def crop_img(self,*roi):
        """ crop_img(x1,y1,x2,y2)
        Returns:
            cropped image
        """
        try:
            return self._img[roi[1]:roi[3],roi[0]:roi[2]]
        except Exception as e:
            if str(e)=="tuple index out of range":
                print("pass coordinates x1,y1,x2,y2 to crop image")
            else:
                print("set image first using : set_image(img)")
            print(e)
            sys.exit(0)

    def resize_image(self,factor):
        h,w = self.shape()[:2]
        print("HW OF GARY IMG==================",h,w)
        return cv2.resize(self._img, (int(w/factor), int(h/factor)))


    def histogram(self):
        pass

    def hsv_values(self,img):
        Selected = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = np.amin(np.extract(Selected[:, :, 0] > 0, Selected[:, :, 0]))
        s_min = np.amin(np.extract(Selected[:, :, 1] > 0, Selected[:, :, 1]))
        v_min = np.amin(np.extract(Selected[:, :, 2] > 0, Selected[:, :, 2]))

        h_max = np.amax(Selected[:, :, 0])
        s_max = np.amax(Selected[:, :, 1])
        v_max = np.amax(Selected[:, :, 2])
        return (h_min,s_min,v_min),(h_max,s_max,v_max)

    def threshold_value(self,img):
        min = np.amin(np.extract(img[:] > 0, img[:]))
        max= np.amax(img[:])
        return min,max

# if __name__=="__main__":
#     img=cv2.imread("org.jpg",0)
#     # vs=VideoStream("acA1920-40uc",True)
#     im=Image(img)
#     im.shape()
#     hsv=im.segmentation(127,100,0)
#     area=im.area(0,0)
#     # crop=im.crop_img(50,50,150,150)
#     print(area)
#     # cv2.imshow("hsv",hsv)
#     # cv2.waitKey(0)
