import cv2
import sys
sys.path.append('hough_transform')
import gh
import math
import numpy as np
import pytesseract
from image_class import *
# from basler_cam import VideoStream
import pyzbar.pyzbar as pyzbar

class Operations(Image):
    """
    Methods:
        color_match()
        dimension_match()
        template_match()
        find_QR_barcode()
        find_character()
    """
    def __init__(self):

        self._temp_image=""
        self.camera_type="color"
        super().__init__()
        self.center_list=[(0,0),(0,0)]

    def set_camera_type(self,cam_type):
        self.camera_type=cam_type

    def set_source_image(self,img):
        self._source_image=img

    def set_temp_image(self,img):
        self._temp_image=img

    def __distance(self,x,y,cx,cy):
        print("IN __distance----------------------")
        x = np.power(cx - x, 2)
        y = np.power(cy - y, 2)
        return np.sqrt(x + y)

    def color_match(self,*hsv):
        """
        color_match(list_of_hsv_value)

        Returns:
            area of big contour.
        """
        try:
            self._mask = self.segmentation(*hsv)
            wpc = self.area()
            return wpc
        except Exception as e:
            print(e)

    def dimension_match(self,*data):
        """
        dimension_match(temp_img,cx1,cy1,cx2,cy2,h_min,s_min,v_min,h_max,s_max,v_max,length)
        where:
            s_img = source image
            t_img = template image
            cx1,cy1,cx2,cy2 = line coordinates(x1,y1,x2 y2)
            .
            .
            length = line length

        Returns:
            new line coordinates (x1,x2,y1,y2) , length ,status -> good or bad
        """
        try:
            T =data[0]
            # template = cv2.cvtColor(data[0],cv2.COLOR_BGR2HSV)
            # t_gray=cv2.cvtColor(T,cv2.COLOR_BGR2GRAY)
            #
            # I = self._img
            # i_gray=cv2.cvtColor(self._img,cv2.COLOR_BGR2GRAY)
            # image = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)

            #find object in image
            w, h = T.shape[:2]
            print("WH:",w,h)
            # h1,w1=I.shape[:2]
            #
            # x=w1/h1
            # print("RATIO::",w1,h1,x)
            # print(data)

            # pos = gh.Gen_Hough(i_gray, t_gray,False,100,360,30)
            pos=self.template_match(data[0])
            # print(pos)
            pos_list=[]
            print("POS LIST==============",pos)
            if len(pos)>0:
                print("IN POS TRUE")

                for pt in pos:
                    x1=pt[0]-w/2
                    y1=pt[1]-h/2
                    x2=pt[0]+w/2
                    y2=pt[1]+h/2
                    # print(x1,y1,x2,y2)

                    coor_list=[]

                    # co_x=[int(data[2]),int(data[4])]
                    # co_y=[int(data[3]),int(data[5])]

                    pix1=np.array([int(data[1]),int(data[2])]).reshape(1,2)
                    pix2=np.array([int(data[3]),int(data[4])]).reshape(1,2)
                    # print("pix1_property:::::::::::::::::::::::::::::::::::;",pix1,pix1.shape)
                    # print("pix2_property:::::::::::::::::::::::::::::::::::;",pix2,pix2.shape)

                    # print("CO_X:",co_x,co_y)
                    #crop selected image
                    crop_img = self._img[int(abs(y1)):int(abs(y2)), int(abs(x1)):int(abs(x2))]

                    print("SIze of CROPED IMage::",crop_img.shape)
                    #apply threashold on croped IMage
                    if self.camera_type=="gray":
                        crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                        mask = cv2.inRange(crop_img,int(data[5]),int(data[6]))
                        cv2.imwrite("static/mmsskk.jpg",mask)
                    else:
                        mask = cv2.inRange(crop_img, (int(data[5]),int(data[6]),int(data[7])), (int(data[8]),int(data[9]),int(data[10])))

                    # print("MASK:" )
                    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    # print("CONUTR77777777777777777777777777777777777777::",contours)

                    c_list=[]
                    for cnt in contours:
                        for c in cnt:
                            c_list.append([c[0][0],c[0][1]])

                    print("cList===",len(c_list))
                    # for i in range(len(co_x)):
                    #     d=[]
                    #     for j in range(len(c_list)):
                    #         d.append(self.__distance(co_x[i],co_y[i],c_list[j][0],c_list[j][1]))# need to optimize
                    #
                    #     min_d=np.argmin(np.array(d))
                    #
                    #     coor_list.append(min_d)
                    # print("coor_list====",coor_list)
                    try:
                        contr_array=np.array(c_list)
                        # print("contr_array_property:::::::::::::::::::::::::::::::::::;",contr_array,contr_array.shape)
                        contr_array_pix1=contr_array-pix1
                        # new_index=np.argmin(np.sqrt(np.sum(contr_array_pix1**2,axis=1)))
                        new_index=(np.sqrt((contr_array_pix1**2).sum(axis=1))).argmin()
                        # print("new index:::::::::::::",new_index)
                        coor_list.append(new_index)

                        contr_array_pix2=contr_array-pix2
                        new_index2=(np.sqrt((contr_array_pix2**2).sum(axis=1))).argmin()
                        # print("new index2:::::::::::::",new_index2)
                        # print(c_list)
                        coor_list.append(new_index2)

                        # print("coor_list====",coor_list)
                        px1=c_list[coor_list[0]][0]+int(x1)
                        py1=c_list[coor_list[0]][1]+int(y1)

                        px2=c_list[coor_list[1]][0]+int(x1)
                        py2=c_list[coor_list[1]][1]+int(y1)
                        # print("pix val===",px1,px2,py1,py2)
                        # print("pix from config======",pix1,pix2)
                        dist=self.__distance(int(px1),int(py1),int(px2),int(py2))
                        print("DISTANCE----------",dist)
                        if self.camera_type=="gray":
                            config_length=data[7]
                        else:
                            config_length=data[11]
                        if int(config_length)-3<=int(dist)<=int(config_length)+3:
                            print("In DIST GOOD")
                            pos_list.append([px1,px2,py1,py2,dist,(0,255,0),"good"])
                        else:
                            print("IN DIST BAD")
                            pos_list.append([px1,px2,py1,py2,dist,(0,0,255),"bad"])
                    except:
                        print("IN EXCEPTtttttt")
                        pos_list=[]


            else:
                print("In POS else")

            # print("POS LIST|:::",)
            return pos_list

        except Exception as e:
            print(e)


    def template_match(self,temp_img):
        """
        template_match(template_img)
        Returns:
            location where the match is found in the source image.
        """
        try:
            img_gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            try:
                w, h = template.shape[:2]
            except:
                w,h=0,0
            # print("HW:::>>>>>>>>>",w,h,img_gray.shape)
            pos = gh.Gen_Hough(img_gray, template,False,100,360,100)
            print("len:::================================================================================================",len(pos),pos)
            return pos

        except Exception as e:
            print(e)


    def find_QR_barcode(self):
        """
        find_QR_barcode(image)
        Returns:
            location,type of barcode/Qr code, information
        """
        try:
            type=None
            data=None
            hull=None
            decodedObjects = pyzbar.decode(self._img)

            for decodedObject in decodedObjects:
                points = decodedObject.polygon

            # If the points do not form a quad, find convex hull
                if len(points) > 4 :
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    hull = list(map(tuple, np.squeeze(hull)))
                else :
                    hull = points;

            for obj in decodedObjects:
                # print('Type : ', obj.type)
                # print('Data : ', obj.data,'\n')
                type=obj.type
                data=obj.data
            # print("HULL IN OPERATION FILE__________________________:",hull)
            return hull,type,data

        except Exception as e:
            print(e)


    def find_character(self):
        """
        find_character(image)
        Returns:
            Text
        """
        try:
            # print("find text funxtion",self._img.shape)

            # Define config parameters.
            # '-l eng'  for using the English language
            # '--oem 1' for using LSTM OCR Engine
            config = ('-l eng --oem 1 --psm 3')
            # Run tesseract OCR on image
            text = pytesseract.image_to_string(self._img, config=config)
            # print("Find text done")
            # Print recognized text
            return text

        except Exception as e:
            print(e)


    def track_object_by_contour(self,*data):

        # imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(imghsv, (h_min,s_min,v_min),(h_max,s_max,v_max))
        hsv=data[4:]
        print("HSV value IN TRACK____",hsv,data,data[0])
        print(data[1])
        mask= self.segmentation(hsv[0],hsv[1],hsv[2],hsv[3],hsv[4],hsv[5])
        cv2.imwrite("static/mask.jpg",mask)
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print("Dist Contr Len :",len(contours))
        pix1=np.array([int(data[0]),int(data[1])]).reshape(1,2)
        pix2=np.array([int(data[2]),int(data[3])]).reshape(1,2)
        c_list=[]
        coor_list=[]
        # co_x=[x1,x2]
        # co_y=[y1,y2]
        # print(co_x,co_y)
        for l in range(len(contours)):
            for i in contours[l]:
                c_list.append(i)

        contr_array=np.array(c_list)
        print("contr_array_property:::::::::::::::::::::::::::::::::::;",contr_array,contr_array.shape)
        contr_array_pix1=contr_array-pix1
        # new_index=np.argmin(np.sqrt(np.sum(contr_array_pix1**2,axis=1)))
        new_index=(np.sqrt((contr_array_pix1**2).sum(axis=1))).argmin()
        # print("new index:::::::::::::",new_index)
        coor_list.append(new_index)

        contr_array_pix2=contr_array-pix2
        new_index2=(np.sqrt((contr_array_pix2**2).sum(axis=1))).argmin()
        # print("new index2:::::::::::::",new_index2)
        # print(c_list)
        coor_list.append(new_index2)

        print("coor_list====",c_list[coor_list[0]][0])
        print("coor_list====",c_list[coor_list[1]][0])
        px1=c_list[coor_list[0]][0][0]
        py1=c_list[coor_list[0]][0][1]

        px2=c_list[coor_list[1]][0][0]
        py2=c_list[coor_list[1]][0][1]

        print(px1,py1,px2,py2)

        pix_dist=int(self.__distance(int(px1),int(py1),int(px2),int(py2)))
        print("DISTANCE",pix_dist)

        return ((px1,py1,px2,py2),pix_dist)


    def demo_track(self,*data):
        hsv=data[4:]
        print("HSV value IN TRACK____",hsv,data,data[0])
        print(data[1])
        if len(hsv)>2:
            mask= self.segmentation(hsv[0],hsv[1],hsv[2],hsv[3],hsv[4],hsv[5])
        else:
            mask= self.segmentation(hsv[0],hsv[1],0)

        # _,thresh = cv2.threshold(imghsv,127,255,cv2.THRESH_BINARY_INV)
        # edges = cv2.Canny(imgray, 100, 200)
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        print("Dist Contr Len :",len(contours))



        c_list=[]
        coor_list=[]
        co_x=[data[0],data[2]]
        co_y=[data[1],data[3]]
        print(co_x,co_y)
        for l in range(len(contours)):
            for i in contours[l]:
                c_list.append(i)

        for i in range(len(co_x)):
            d=[]
            for j in range(len(c_list)):
                # d.append(self.distance(x,y,contours[0][i]))
                d.append(self.__distance(co_x[i],co_y[i],c_list[j][0][0],c_list[j][0][1]))

        # print(d)
            min_d=np.argmin(np.array(d))
            coor_list.append(min_d)
            # co_list[0]=co_list[1]
            # co_list[1]=min_d
        print("coor_list",coor_list)

        print(len(c_list))

        px1=c_list[coor_list[0]][0][0]
        py1=c_list[coor_list[0]][0][1]

        px2=c_list[coor_list[1]][0][0]
        py2=c_list[coor_list[1]][0][1]
        self.center_list[0]=(px1,py1)
        self.center_list[1]=(px2,py2)
        pix_dist=int(self.__distance(int(px1),int(py1),int(px2),int(py2)))
        # print("cnt_area",self.cnt_area)
        return ((px1,py1,px2,py2),pix_dist,self.center_list)#,self.cnt_area,self.center_list)#,"%.2f" % pix_dist_mm)
        # print("cnt_area",self.cnt_area)
#
# if __name__=="__main__":
#     # vs=VideoStream("acA1920-40uc",True)
#     simg=cv2.imread("s_img.jpg")
#     timg=cv2.imread("t1.jpg")
#     opr=Operations()
#     loc=opr.template_match(simg,timg)
#     print(loc)
#     w, h = timg.shape[:2]
#     if loc is not None:
#             for pt in loc:
#                 x1=pt[0]-h/2
#                 y1=pt[1]-w/2
#                 x2=pt[0]+h/2
#                 y2=pt[1]+w/2
#                 print(x1,y1,x2,y2)
#                 cv2.rectangle(simg, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
#     cv2.imshow("t",simg)
#     cv2.waitKey(0)
    # while 1:
    #     img=vs.capture()
    #     h,w = img.shape[:2]
    #     resized_image = cv2.resize(img, (int(w/4), int(h/4)))
    #     data=opr.find_QR_barcode(resized_image)
    #
    #     # Draw the convext hull
    #     try:
    #         n = len(data[0])
    #         for j in range(0,n):
    #             cv2.line(resized_image, data[0][j], data[0][ (j+1) % n], (255,0,0), 3)
    #     except:
    #         print("not found")
    #
    #     print(data)
    #     cv2.imshow("frame",resized_image)
    #     cv2.waitKey(1)
