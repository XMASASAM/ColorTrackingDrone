from djitellopy import tello
import numpy as np
import cv2
from enum import Enum

class Move(Enum):
    Up=1
    Down=2
    Left=3
    Right=4
    Straight=5

def drawBox(img,bbox,color):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), color,3)

class Console:#デバック用のクラス
    def __init__(self,name,wx,wy):#windowの名前、幅、高さ
        self.__img = np.zeros((wx,wy,3))
        self.__str = {}
        self.__name = name

    def show(self):#表示
        self.__img[:] = (0,0,0)#初期化（黒で塗りつぶす）
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL#フォントを設定
        if self.__str is None:
            return
        for i in self.__str.values():
            cv2.putText(self.__img,str(i[0]),(i[1],i[2]),font,1,(255,255,255),2,cv2.LINE_4)#文字を描く
        cv2.imshow(str(self.__name),self.__img)#windowを表示

    def write(self,tag,str,x=0,y=0):#描く文字を設定
        self.__str[tag] = (str,x*24,y*24+16)

class Delection:#指定した範囲の色を抽出し、認識させるクラス
    def __init__(self,low,high):#抽出する色（HSV）の最低値、最高値
        self.lower = np.array(low)
        self.upper = np.array(high)
        self.kernel = np.ones((5,5),np.uint8)

    def getCont(self,hsv):#指定された色だけを切り取り、それを四角形で囲む
        mask = cv2.inRange(hsv, self.lower, self.upper)#閾値により白黒の画像が出力される
        img_binary = cv2.dilate(mask,self.kernel,iterations = 10)#線を太くする
        contours, _ = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)#認識した物体のデータを出力
        
        if len(contours)==0:
            return None
        
        return contours

class Operation:#ドローンを操作するクラス
    def __init__(self,screenWH,scope):#画像のサイズ、許容範囲
        self.__width = screenWH[0]
        self.__height = screenWH[1]
        middleX = self.__width >> 1
        middleY = self.__height >> 1
        self.__left = middleX - (scope >> 1)#左回りの基準線
        self.__top = middleY - (scope >> 1)#上昇の基準線
        self.__right = self.__left + scope#右回りの基準線
        self.__bottom = self.__top + scope#下降の基準線
        self.__oldMove = Move.Left


    def Update(self,cont):
        if cont is None:#何も見つからなかった場合
            self.__NoObject=True
            self.__move = self.__oldMove#最後にしていた動作を行う
            return;

        self.__NoObject = False
        self.__bbox = cv2.boundingRect(cont[0])#四角形を１つ出力
        x,y,w,h = self.__bbox
        self.__cX , self.__cY = x + (h >> 1) , y + (w >> 1) #四角形の中心
        
        cX , cY = self.__cX , self.__cY

        self.__move = Move.Straight#認識した物体が画像の真ん中にあると真っ直ぐ進む

        if cX < self.__left:#物体が左側にあったら左回りをする
            self.__move = Move.Left
        elif self.__right < cX:#物体が右側にあったら右回りをする
            self.__move = Move.Right

        if cY < self.__top:#物体が上側にあったら上昇する
            self.__move = Move.Up
        elif self.__bottom < cY:#物体が下側にあったら下降する
            self.__move = Move.Down
        
        self.__oldMove = self.__move
        
        
    def Draw(self,img):
        if self.__NoObject:#何も認識しなかったら何も描かない
            return

        drawBox(img,self.__bbox,(255,0,255))#認識した物体を四角で囲む ピンク

        lcolor = (255,0,0)#青
        _=cv2.line(img,(self.__left,0),(self.__left,self.__height),lcolor,3)#左回りの基準線を描く
        _=cv2.line(img,(self.__right,0),(self.__right,self.__height),lcolor,3)#右回りの基準線を描く
        _=cv2.line(img,(0,self.__top),(self.__width,self.__top),lcolor,3)#上昇の基準線を描く
        _=cv2.line(img,(0,self.__bottom),(self.__width,self.__bottom),lcolor,3)#下降の基準線を描く


        ccolor = (0,255,0)#緑
        _=cv2.circle(img,(self.__cX,self.__cY),16,ccolor,3)#物体を囲った四角形の中心を円で描く


    def getMove(self)->Move:#動作を出力
        return self.__move

    def foundObject(self)->bool:#物体を認識したかを出力
        return not self.__NoObject





def Initialize():#初期化
    global mytello,mask,distance,text,screenWH,controller#グローバル変数として宣言
    mytello = tello.Tello()#ドローン
    mytello.connect()
    print(mytello.get_battery())#電池の容量を出力
    mytello.streamon()
    mytello.takeoff()#離陸

    distance = 30#ドローンが動く単位
    screenWH = (360,240)#ドローンから送られてくる画像をリサイズする大きさ（width,height）
    controller = Operation(screenWH,30)#制御用
   # mask = Delection([123,197,203],[135,255,255])#認識させる色を設定
   # mask = Delection([123,197,161],[140,255,255])#認識させる色を設定
   # mask = Delection([162,78,62],[173,110,161])#認識させる色を設定
    mask = Delection([114, 227, 159],[117, 255, 185])#認識させる色を設定

    text = Console('console',500,500)#デバック用



def Update():
    global img
    key = cv2.waitKey(1)#キーボードからの入力
    if key == ord('q'):#「q」を押すと終了
        mytello.land()#着陸
        return 0#0を返すことにより無限ループが終了
    
    img = mytello.get_frame_read().frame#ドローンから画像を取得
    img = cv2.resize(img,screenWH)#リサイズ
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)#HSVへ変換

    cont = mask.getCont(hsv)#指定されたの色を認識

    controller.Update(cont)#認識したデータを用いて制御

    #デバック用のwindowを初期化
    text.write('y','-')
    text.write('x','-')
    text.write('z','-')

    #ドローンの状態を表示
    if(controller.foundObject()):
        text.write('find','Chasing a object',0,1)
    else:
        text.write('find','Looking for any object',0,1)

    #実際にドローンを動かす
    move = controller.getMove()#どんな操作をするのかを取得
    if move == Move.Up:#上昇
        text.write('y','up',0,0)
        mytello.send_rc_control(0,0,distance,0)
    elif move == Move.Down:#下降
        text.write('y','down',0,0)
        mytello.send_rc_control(0,0,-distance,0)

    if move == Move.Left:#左回り
        text.write('x','left',4,0)
        mytello.send_rc_control(0,0,0,-distance)
    elif move == Move.Right:#右回り
        text.write('x','right',4,0)
        mytello.send_rc_control(0,0,0,distance)
    
    if move == Move.Straight:#真っ直ぐ
        text.write('z','straight',8,0)
        mytello.send_rc_control(0,distance,0,0)

    return 1#1を返すことによりプログラムを続行

def Draw():
    text.show()#デバック用のwindowが表示
    controller.Draw(img)#認識している様子を描く
    cv2.imshow('image',img)#ドローンからの画像（加工済み）を表示


if __name__ == '__main__':
    Initialize()#初期化

    #メインループ
    while Update():
        Draw()
    
    cv2.destroyAllWindows()#全てのwindowを削除して終了


