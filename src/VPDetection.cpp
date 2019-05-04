#include "VPDetection.h"
#include "time.h"
#include <iostream>

using namespace std;
using namespace cv;


VPDetection::VPDetection(void)
{
}


VPDetection::~VPDetection(void)
{
}

void VPDetection::run( std::vector<std::vector<double> > &lines, cv::Point2d pp, double f, std::vector<cv::Point3d> &vps, std::vector<std::vector<int> > &clusters )
{
	this->lines = lines;
	this->pp = pp;
	this->f = f;
    this->noiseRatio = 0.5;//假设不在主方向上直线比率

	cout<<"get vp hypotheses . . ."<<endl;
	std::vector<std::vector<cv::Point3d> > vpHypo;
	getVPHypVia2Lines( vpHypo );

	cout<<"get sphere grid . . ."<<endl;
	std::vector<std::vector<double> > sphereGrid;
	getSphereGrids( sphereGrid );

	cout<<"test vp hypotheses . . ."<<endl;
	getBestVpsHyp( sphereGrid, vpHypo, vps );
    //得到最佳的消失点赋给vps


	cout<<"get final line clusters . . ."<<endl;
    //得到归一化的灭点后找各组直线
    double thAngle = 6.0 / 180.0 * CV_PI;//这个参数是直线与灭点接近的阈值
	lines2Vps( thAngle, vps, clusters );
	int clusteredNum = 0;
	for ( int i=0; i<3; ++i )
	{
		clusteredNum += clusters[i].size();
	}

	cout<<"total: " <<lines.size()<<"  clusered: "<<clusteredNum;
	cout<<"   X: "<<clusters[0].size()<<"   Y: "<<clusters[1].size()<<"   Z: "<<clusters[2].size()<<endl;
}

void VPDetection::getVPHypVia2Lines( std::vector<std::vector<cv::Point3d> > &vpHypo )//通过两条线计算灭点
{
	int num = lines.size();

    double p = 1.0 / 3.0 * pow( 1.0 - noiseRatio, 2 );//初始假设的随机选择的两条直线为同一个主方向的灭点概率最小1/3

    double confEfficience = 0.9999;//置信度
    int it = log( 1 - confEfficience ) / log( 1.0 - p );//计算灭点1的迭代次数
	
	int numVp2 = 360;
    double stepVp2 = 2.0 * CV_PI / numVp2;//假设的灭点2所在圆的单位长度，到时生成灭点2是一度一单位，所以一个灭点1会生成360个灭点2;

	// get the parameters of each line
	lineInfos.resize( num );
	for ( int i=0; i<num; ++i )
	{
		cv::Mat_<double> p1 = ( cv::Mat_<double>(3, 1) << lines[i][0], lines[i][1], 1.0 );
		cv::Mat_<double> p2 = ( cv::Mat_<double>(3, 1) << lines[i][2], lines[i][3], 1.0 );

        lineInfos[i].para = p1.cross( p2 );//储存线条向量

		double dx = lines[i][0] - lines[i][2];
		double dy = lines[i][1] - lines[i][3];
        lineInfos[i].length = sqrt( dx * dx + dy * dy );//储存线条长度

        lineInfos[i].orientation = atan2( dy, dx );//储存线条方向
		if ( lineInfos[i].orientation < 0 )
		{
			lineInfos[i].orientation += CV_PI;
		}
	}

	// get vp hypothesis for each iteration
    vpHypo = std::vector<std::vector<cv::Point3d> > ( it * numVp2, std::vector<cv::Point3d>(3)  );//储存灭点2
	int count = 0;
	srand((unsigned)time(NULL));  
	for ( int i = 0; i < it; ++ i )
	{
        //选择线条，这里灭点1的随机生成机制可以考虑改变，这里是是根据概率学，至少选择it组的两个不同的线条会算出一个灭点;
		int idx1 = rand() % num;
		int idx2 = rand() % num;
		while ( idx2 == idx1 )
		{
			idx2 = rand() % num;
		}

		// get the vp1
		cv::Mat_<double> vp1_Img = lineInfos[idx1].para.cross( lineInfos[idx2].para );
        if ( vp1_Img(2) == 0 )//无限灭点排除
		{
			i --;
			continue;
		}
        //灭点1的三维球体坐标
		cv::Mat_<double> vp1 = ( cv::Mat_<double>(3, 1) << vp1_Img(0) / vp1_Img(2) - pp.x, vp1_Img(1) / vp1_Img(2) - pp.y, f );
        if ( vp1(2) == 0 ) { vp1(2) = 0.0011; }//避免焦距为0
		double N = sqrt( vp1(0) * vp1(0) + vp1(1) * vp1(1) + vp1(2) * vp1(2) );
        vp1 *= 1.0 / N;//归一化

        //遍历生成 get the vp2 and vp3
		cv::Mat_<double> vp2 = ( cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0 );
		cv::Mat_<double> vp3 = ( cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0 );
		for ( int j = 0; j < numVp2; ++ j )
		{
			// vp2
			double lambda = j * stepVp2;

			double k1 = vp1(0) * sin( lambda ) + vp1(1) * cos( lambda );
			double k2 = vp1(2);
			double phi = atan( - k2 / k1 );

			double Z = cos( phi );
			double X = sin( phi ) * sin( lambda );
			double Y = sin( phi ) * cos( lambda );

			vp2(0) = X;  vp2(1) = Y;  vp2(2) = Z;
			if ( vp2(2) == 0.0 ) { vp2(2) = 0.0011; }
			N = sqrt( vp2(0) * vp2(0) + vp2(1) * vp2(1) + vp2(2) * vp2(2) );
			vp2 *= 1.0 / N;
			if ( vp2(2) < 0 ) { vp2 *= -1.0; }		

			// vp3
			vp3 = vp1.cross( vp2 );
			if ( vp3(2) == 0.0 ) { vp3(2) = 0.0011; }
			N = sqrt( vp3(0) * vp3(0) + vp3(1) * vp3(1) + vp3(2) * vp3(2) );
			vp3 *= 1.0 / N;
			if ( vp3(2) < 0 ) { vp3 *= -1.0; }		

			//
			vpHypo[count][0] = cv::Point3d( vp1(0), vp1(1), vp1(2) );
			vpHypo[count][1] = cv::Point3d( vp2(0), vp2(1), vp2(2) );
			vpHypo[count][2] = cv::Point3d( vp3(0), vp3(1), vp3(2) );

			count ++;
		}
	}
}


void VPDetection::getSphereGrids( std::vector<std::vector<double> > &sphereGrid )
{	
	// build sphere grid with 1 degree accuracy
    double angelAccuracy = 1.0 / 180.0 * CV_PI;//1度
    double angleSpanLA = CV_PI / 2.0;//纬度0-90
    double angleSpanLO = CV_PI * 2.0;//经度0-360
	int gridLA = angleSpanLA / angelAccuracy;
	int gridLO = angleSpanLO / angelAccuracy;
    //std::cout<<"start!"<<std::endl;
    std::vector<double> sphereGrid_unit;
    //初始化
    sphereGrid_unit.resize(gridLO);    
    for(int i = 0; i< gridLO;i++)
        sphereGrid_unit[i] = 0.0;

	for ( int i=0; i<gridLA; ++i )
	{
        sphereGrid.push_back(sphereGrid_unit);
	}

    // put intersection points into the grid
	double angelTolerance = 60.0 / 180.0 * CV_PI;
	cv::Mat_<double> ptIntersect;
	double x = 0.0, y = 0.0;
	double X = 0.0, Y = 0.0, Z = 0.0, N = 0.0;
	double latitude = 0.0, longitude = 0.0;
	int LA = 0, LO = 0;
	double angleDev = 0.0;
	for ( int i=0; i<lines.size()-1; ++i )
	{
		for ( int j=i+1; j<lines.size(); ++j )
		{
			ptIntersect = lineInfos[i].para.cross( lineInfos[j].para );

			if ( ptIntersect(2,0) == 0 )
			{
				continue;
			}

			x = ptIntersect(0,0) / ptIntersect(2,0);
			y = ptIntersect(1,0) / ptIntersect(2,0);

			X = x - pp.x;
			Y = y - pp.y;
			Z = f;
			N = sqrt( X * X + Y * Y + Z * Z );

			latitude = acos( Z / N );
			longitude = atan2( X, Y ) + CV_PI;

			LA = int( latitude / angelAccuracy );
			if ( LA >= gridLA ) 
			{
				LA = gridLA - 1;
			}

			LO = int( longitude / angelAccuracy );
			if ( LO >= gridLO ) 
			{
				LO = gridLO - 1;
			}

			// 
			angleDev = abs( lineInfos[i].orientation - lineInfos[j].orientation );
			angleDev = min( CV_PI - angleDev, angleDev );
			if ( angleDev > angelTolerance )
			{
				continue;
			}
            //相当于将每两个线条的交点放到对应的格子中，并计算每个格子储存的是一个衡量放进来交点数量和质量的参数
			sphereGrid[LA][LO] += sqrt( lineInfos[i].length * lineInfos[j].length ) * ( sin( 2.0 * angleDev ) + 0.2 ); // 0.2 is much robuster
		}
	}

	// 
    int halfSize = 1;//1
    int winSize = halfSize * 2 + 1;//3
    int neighNum = winSize * winSize;//9

    /************get the weighted line length of each grid***3X3高斯滤波原来的网格********************/

    //初始化用来高斯处理的新网格
    std::vector<std::vector<double> > sphereGridNew;

    for(int i = 0; i< gridLO;i++)
        sphereGrid_unit[i] = 0.0;

    //sphereGridNew.resize(gridLA);
    for ( int i=0; i<gridLA; ++i )
    {
        sphereGridNew.push_back(sphereGrid_unit);
    }


    //实际就是得到每格网格周围3x3的权重/9后，与与原网格权重相加，可以避免有些网格没有数据
	for ( int i=halfSize; i<gridLA-halfSize; ++i )
	{
		for ( int j=halfSize; j<gridLO-halfSize; ++j )
		{
			double neighborTotal = 0.0;
			for ( int m=0; m<winSize; ++m )
			{
				for ( int n=0; n<winSize; ++n )
				{
					neighborTotal += sphereGrid[i-halfSize+m][j-halfSize+n];
				}
			}

			sphereGridNew[i][j] = sphereGrid[i][j] + neighborTotal / neighNum;
		}
	}
	sphereGrid = sphereGridNew;
}

void VPDetection::getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<cv::Point3d> > &vpHypo, std::vector<cv::Point3d> &vps )
{
	int num = vpHypo.size();
	double oneDegree = 1.0 / 180.0 * CV_PI;

	// get the corresponding line length of every hypotheses
	std::vector<double> lineLength( num, 0.0 );

    for ( int i = 0; i < num; ++ i )
	{
		std::vector<cv::Point2d> vpLALO( 3 ); 
		for ( int j = 0; j < 3; ++ j )
		{
            //去掉归一化失败的灭点，其实可以不要，因为生成这些假设点时就已经去掉了
            if ( vpHypo[i][j].z == 0.0 )
            {
                continue;
            }

            if ( vpHypo[i][j].z > 1.0 || vpHypo[i][j].z < -1.0 )
            {
                cout<<1.0000<<endl;
            }


            //计算每个灭点经纬度
			double latitude = acos( vpHypo[i][j].z );
			double longitude = atan2( vpHypo[i][j].x, vpHypo[i][j].y ) + CV_PI;

			int gridLA = int( latitude / oneDegree );
			if ( gridLA == 90 ) 
			{
				gridLA = 89;
			}
			
			int gridLO = int( longitude / oneDegree );
			if ( gridLO == 360 ) 
			{
				gridLO = 359;
			}
            //计算每组假设的3个灭点权重
			lineLength[i] += sphereGrid[gridLA][gridLO];
		}
	}

	// get the best hypotheses
	int bestIdx = 0;
	double maxLength = 0.0;
	for ( int i = 0; i < num; ++ i )
	{
		if ( lineLength[i] > maxLength )
		{
			maxLength = lineLength[i];
			bestIdx = i;
		}
	}
    //选择权重最好的灭点组作为最佳灭点
	vps = vpHypo[bestIdx];
}


void VPDetection::lines2Vps( double thAngle, std::vector<cv::Point3d> &vps, std::vector<std::vector<int> > &clusters )
{
	clusters.clear();
	clusters.resize( 3 );
//实际就是判断3个灭点与直线段中点连线和直线角度是否接近，判断直线属于哪个方向或者属于其他方向
    //get the corresponding vanish points on the image plane  由焦点和焦距恢复灭点的像素坐标
	std::vector<cv::Point2d> vp2D( 3 ); 
	for ( int i = 0; i < 3; ++ i )
	{
		vp2D[i].x =  vps[i].x * f / vps[i].z + pp.x;
		vp2D[i].y =  vps[i].y * f / vps[i].z + pp.y;
	}


	for ( int i = 0; i < lines.size(); ++ i )
	{
		double x1 = lines[i][0];
		double y1 = lines[i][1];
		double x2 = lines[i][2];
		double y2 = lines[i][3];
		double xm = ( x1 + x2 ) / 2.0;
		double ym = ( y1 + y2 ) / 2.0;

		double v1x = x1 - x2;
		double v1y = y1 - y2;
		double N1 = sqrt( v1x * v1x + v1y * v1y );
        v1x /= N1;   v1y /= N1;//每条直线x，y的角度

		double minAngle = 1000.0;
		int bestIdx = 0;
		for ( int j = 0; j < 3; ++ j )
		{
			double v2x = vp2D[j].x - xm;
			double v2y = vp2D[j].y - ym;
			double N2 = sqrt( v2x * v2x + v2y * v2y );
            v2x /= N2;  v2y /= N2;//灭点与直线段中点连线的x，y方向角度

			double crossValue = v1x * v2x + v1y * v2y;
			if ( crossValue > 1.0 )
			{
				crossValue = 1.0;
			}
			if ( crossValue < -1.0 )
			{
				crossValue = -1.0;
			}
			double angle = acos( crossValue );
			angle = min( CV_PI - angle, angle );

			if ( angle < minAngle )
			{
				minAngle = angle;
				bestIdx = j;
			}
		}

		//
		if ( minAngle < thAngle )
		{
            clusters[bestIdx].push_back( i );//推入的是直线序号
		}
	}
}
