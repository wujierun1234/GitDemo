#define _CRT_SECURE_NO_WARNINGS
#include <graphics.h>		// 引用图形库头文件
#include <conio.h>
 
#define NUM 15
#define WIN_NUM 5//五子棋
 
int pieceArr[NUM][NUM] = { 0 };//记录15*15个棋盘的棋子情况,0表示没有棋子,1表示黑子,2表示白子
 
//画线
void Draw_line()
 
//画点
void Draw_point()
 
//判断这个位置是否有其他棋子
bool NicePos(int x, int y)
 
//判断游戏胜利
int GameOver(int x, int y)
 
int main()
{
	initgraph(600,600);	// 创建绘图窗口，大小为 640x480 像素
	
	// 读取图片至绘图窗口
	loadimage(NULL, _T("2.png"));
 
	//画线
	Draw_line();
	
	//画点
	Draw_point();
 
	//鼠标消息
	ExMessage m; //结构体m ,用与存放鼠标消息
 
	//黑子先下
	bool black = true;//黑子先下
 
	_getch();				// 按任意键继续
	closegraph();			// 关闭绘图窗口
	return 0;
}
