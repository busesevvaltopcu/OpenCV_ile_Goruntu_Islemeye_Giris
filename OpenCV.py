import cv2 as cv
import numpy as np

img= cv.imread("images/general/miuul.png")
type(img)

#pencere seçilir
cv.namedWindow("miuul", cv.window_autosize)
cv.imshow("miuul", img)

#pencereyi açık tutma
cv.waitKey(0)

#tuşa basmadan console kullanılabilir hale gelir
cv.waitKey(1)

#pencere kapatılır
cv.destroyAllWindows()

#amacımız yüklediğimiz resmi griye çevirmek olsun:
img_opencv=cv.imread("images/general/opencv.png")
cv.namedWindow("opencv_colored", cv.WINDOW_AUTOSIZE)
cv.imshow("opencv_colored", img_opencv)
cv.waitKey(1)
cv.destroyAllWindows()

gray=cv.cvtColor(img_opencv, cv.COLOR_BGR2GRAY)
cv.imageshow("opencv_gray", gray)
cv.waitKey(1)

cv.imwrite("images/general/opencv.png",gray)

#merging two images
img1 = cv.imread ("images/02/leftimage.jpeg")
img2 = cv.imread ("images/02/rightimage.jpeg")

cv.imshow("spider_man1", img1)
cv.waitkey(0)

cv.imshow("spider_man2", img2)
cv.waitKey(0)

#iki resmi yatay olarak birleştir
horizontal = np.hstack((img1,img2))
cv.imshow("spider_man", horizontal)
cv.waitKey(0)

#pixel read and write
img_opencv=cv.imread("images/general/opencv.png")
cv.namedWindow("opencv_colored", cv.WINDOW_AUTOSIZE)
cv.imshow("opencv_colored", img_opencv)
cv.waitKey(1)

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(1)
    # cv.destroyAllWindows()

show_image(img_opencv)

#seçilen pikseli siyaha boyama
h, w, ch = img_opencv.shape
print("h, w, ch", h, w, ch)

x_start, y_start = 300, 200
x_end, y_end = 500, 400

select_region = img_opencv[y_start:y_end, x_start:x_end]
show_image(select_region)

new_values = [0, 0, 0]

img_opencv2 = np.copy(img_opencv)
img_opencv2[y_start:y_end, x_start:x_end]=new_values
show_image(img_opencv2)
#####

new_img1 = np.zeros_like(img_opencv2.shape, img_opencv.dtype)
show_image(new_img1)

new_img2 = np.zeros([512,512], np.uint8)
show_image(new_img2)

#Griye boyama
new_img3 = np.zeros([512,512], np.uint8)
new_img3[:,:] = 127
show_image(new_img3)
###############

#Resimdeki renkleri tersine çevirme

show_image(img_opencv)
for row in range (h):
    for col in range(w):
        #Her pikselin mavi (b), yeşil (g) ve kırmızı (r) değerlerini al
        b, g, r =img_opencv[row,col]

        #her renk bileşenini 255'ten çıkararak renkleri tersine çevir
        b = 255 - b
        g = 255 - g
        r = 255 - r

        #tersine çevrilen renkleri tekrar resme yerleştir
        img_opencv[row,col] = [b,g,r]

show_image(img_opencv)

#harikaresimcizerim.com yazılı bir görsel oluşturacağız
#800x600 boyutunda siyah bir resim oluştur ve ortasında mavi çizgi çiz
img= np.zeros((800,600,3), detypye=np.uint8)

start_point= (100,100)
end_point = (500,500)
color= (255,0,0) #Mavi renk
thickness = 2
cv.line(img,start_point,end_point, color, thickness)

show_image(img)

#Buse ismini çiz
font=cv.FONT_HERSHEY_SIMPLEX
font_scale=2
font_thickness=3
text="Buse"
text_size = cv.getTextSize(text,font,font_scale,font_thickness)[0]

#ismi resmin ortasına yerleştir
x= (img.shape[1]-text_size[0])//2
y= (img.shape[0]+text_size[1])//2
cv.putText(img, text, (x, y), font, font_scale, (0,0,0), thickness, cv.LINE_AA)
show_image(img)

#550/770 boyutlarında üç kanallı (rgb) beyaz bir resim oluştur (img)
img=np.ones((550,770,3))

#renk tanımlamaları
black= (0,0,0)
red= (0,0,255)
green = (0,255,0)

#kareler ve çizgiler çiz
cv.rectangle(img, (480,250), (100,450), black, 8)
cv.rectangle(img, (580,150), (200,350), black, 8)
cv.line(img, (100,450), (200,350), black, 8)
cv.line(img, (480,450), (580,150), black, 8)
cv.line(img, (100,250), (200,150), black, 8)
cv.line(img, (480,450), (580,350), black, 8)

#metin ekleme
start_point=(150,100)
font_thickness=2
font_size=1
font=cv.FONT_HERSHEY_DUPLEX
cv.putText(img, text, start_point, font, font_size,black, font_thickness)
return img


#streamlit uygulamasını oluştur
st.title("görseli oluşturma")

#kullanıcıdan metin girişi al
user.input = st.text_input("metni girin:", value="www.harikabirresim.com")

#çizilmiş resmi göster
if user.input:
    st.image(draw_image_with_text(user_input), channels= "BGR", use_column_width=True)
else:
    st.warning("lütfen bir metin girin.")
#uygulamayı çalıştırmak için terminalden harikabirresim.py dosyasının olduğu dizine gidin.
#streamlit run harikabirresim.py kodunu çalıştırın.

#################################
# PIXEL ARITHMETIC OPERATIONS
#################################

import cv2 as cv
import numpy as np

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(1)
    cv.destroyAllWindows()

src1= cv.imread ("images/02/test0.jpeg")
src2= cv.imread ("images/02/test1.jpeg")

h, w, ch = src1.shape
print("h, w, ch", h, w, ch)

#add
add_result= np.zeros(src1.shape, src1.dtype)
cv.add(src1, src2, add_result)
show_image(add_result)

#subtract
sub_result= np.zeros(src1.shape, src1.dtype)
cv.subtract(src1, src2, sub_result)
show_image(sub_result)

#mutliply
mul_result= np.zeros(src1.shape, src1.dtype)
cv.multiply(src1, src2, mul_result)
show_image(mul_result)

#divide
div_result= np.zeros(src1.shape, src1.dtype)
cv.divide(src1, src2, div_result)
show_image(div_result)

########################################

img= cv.imread("images/limhann2.jpeg")
show_image(img)

resized_img = cv.resize(img, (800,500))
show_image(resized_img)

high_contrast_img= cv.addWeighted(resized_img, 2,np.zeros_like(resized_img),0 ,0)
show_image(high_contrast_img)

#############################
# IMAGE PSEUDO COLOR ENHANCEMENT
###############################

import cv2 as cv
import numpy as np
import requests

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(1)
   # cv.destroyAllWindows()

src = cv.imread("images/general/limhann2.jpg")
src= cv.resize(src, (800,500))
show_image(src)

dst=cv.applyColorMap(src, cv.COLORMAP_PINK)
show_image(dst)

dst=cv.applyColorMap(src, cv.COLORMAP_AUTUMN)
show_image(dst)

dst=cv.applyColorMap(src, cv.COLORMAP_WINTER)
show_image(dst)

#COLORMAP_AUTUMN
#COLORMAP_BONE
#COLORMAP_WINTER
#COLORMAP_OCEAN
#COLORMAP_SUMMER
#COLORMAP_PINK
#COLORMAP_COOL
#COLORMAP_JET

src=cv.imread("images/general/limhann1.jpg")
src= cv.resize(src, (800,   500 ))

bw_image = cv.cvtColor (src, cv.COLOR_BGR2GRAY)
show_image(bw_image)

#görseli gri okuma
img_gray =cv.imread("images/general/limhann1.jpg", cv.IMREAD_GRAYSCALE)
img_gray = cv.resize(img_gray, (800,500))
show_image(img_gray)

#gri resmi renklendirme
color_map = cv.COLORMAP_JET
img_pseudo_color = cv. applyColorMap(img_gray, color_map)
show_image(img_pseudo_color)
show_image(img_gray)