// pdde_final.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "mul.h"
#define ITER 20
#define START 0
int MAX_FES = 700;            //����      
char _filename[100] = "pic\\person_303.png";  //ͼƬѡ��
char g_filename[][68] = {
	//"pos\\person_265.png",
	//"pos\\crop001676.png",
	//"pic\\1626.jpg",
	//"pic\\1815.jpg",
	"pos\\(1).png",
	"pos\\(2).png",
	"pos\\(3).png",
	"pos\\(4).png",
	"pos\\(5).png",
	"pos\\(6).png",
	"pos\\(7).png",
	"pos\\(8).png",
	"pos\\(9).png",
	"pos\\(10).png",

	"pos\\(11).png",
	"pos\\(12).png",
	"pos\\(13).png",
	"pos\\(14).png",
	"pos\\(15).png", 
	"pos\\(16).png",
	"pos\\(17).png",
	"pos\\(18).png",
	"pos\\(19).png",
	"pos\\(20).png",

	"pos\\(21).png",
	"pos\\(22).png",
	"pos\\(23).png",
	"pos\\(24).png",
	"pos\\(25).png",
	"pos\\(26).png",
	"pos\\(27).png",
	"pos\\(28).png",
	"pos\\(29).png",
	"pos\\(30).png",

	"pos\\(31).png",
	"pos\\(32).png",
	"pos\\(33).png",
	"pos\\(34).png",
	"pos\\(35).png",
	"pos\\(36).png",
	"pos\\(37).png",
	"pos\\(38).png",
	"pos\\(39).png",
	"pos\\(40).png",
	"pos\\(41).png",
	"pos\\(42).png",
	//"pos\\(43).png",
	//"pos\\(44).png",
	//"pos\\(45).png",	//"pos\\(40).png",
	//"pos\\(46).png",
	//"pos\\(47).png",
	//"pos\\(48).png",
	//"pos\\(49).png",
	//"pos\\one(1).png",
	//"pos\\one(2).png",
	//"pos\\one.png",

};

double upper[3] = { 320, 320, 60 };       //ͼƬ��Ĭ��size�Լ�winsize
double lower[3] = { 0, 0, 20 };
const double size_factor = 0.80;            //to resize the rect   0.8width��0.86heightis standrad
const double haar_factor = 0.8;
const double radius_factor = 0.4;         //3.2 is half of the window
const double threshold_hog = -0.5;

const double alpha = 1;         //size factor
const double beta = 0.5;

int nvars = 3;   //��������ά��
Mat img_origin, img, img_for_multiscale, img_for_NSDE, img_for_LAMSACO, img_for_LMCEDA;
int g_fes = 0;
const double stepx = 8;
const double stepy = 8;
int shift = 0;
//const double stepz=0.1;
const double stepz = 2;


int trials = 1;
FILE * outf;
vector<Mat> scaled_pics;
double weight[800][800][60];
float block_des[36][600][500][36] = { 0 };
//float**** block_des;
//HOGDescriptor hog;
//HOG_NSDE hog; 
struct Solution
{
	//double x[4];
	int x[3];      //try the bin DE
	double fitness;
};

#include "defultpeople.h"
#include "func.h"



int readImg(char * fname)
{
	img_for_multiscale = imread(fname);
	img = imread(fname);
	img_for_NSDE = imread(fname);
	img_for_LAMSACO = imread(fname);
	img_for_LMCEDA = imread(fname);
	img_origin = imread(fname);

	if (img.empty()){
		fprintf(stderr, "cannot read test image");
		return 1;
	}
	upper[0] = img.cols;
	upper[1] = img.rows;
	//if(upper[2]>img.rows/13)
	upper[2] = img.rows * int(X2_SCALE) / 128;        //����128�ٳ�10
	upper[2] = cvCeil(upper[2] * alpha);                  // 1.5,1.5 for small

	lower[2] = cvCeil(upper[2] * beta);
	cout << "size of pic:" << img_origin.size() << endl;

	return 0;
}


void object_function(struct Solution *sol)
{
	g_fes++;
	sol->fitness = -1 * funceval(sol->x);
}


/************************************************************************
* Function Name : bounding_solution
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : ��ֹԽ�磬x[2] �����x[1] x[0] �����½�
*
* Param : solutions: ָ�������NPά solution ����
**
************************************************************************/
void bounding_solution(struct Solution *s)
{
	int k;
	for (k = 0; k < Dim; k++)
	if (s->x[k] < lower[k])
		s->x[k] = (int)lower[k];

	if (upper[2] < s->x[2])
		s->x[2] = (int)upper[2];

	if ((upper[0] - 64 * (s->x[2] / X2_SCALE) + shift) < s->x[0])             //X��Y���������޻���winsize�ı仯���仯
		s->x[0] = (int)(upper[0] - 64 * (s->x[2] / X2_SCALE) + shift);       // 10->5
	if ((upper[1] - 128 * (s->x[2] / X2_SCALE) + shift) < s->x[1])
		s->x[1] = (int)(upper[1] - 128 * (s->x[2] / X2_SCALE) + shift);     // 10->5.0
}


/************************************************************************
* Function Name : initialize_NP_solutions
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : ��ʼ�� NP ���⣬������ʱʹ��ԭ�����randnum������ʹ�� boots �⣡ 
*
* Param : solutions: ָ�������NPά solution ����
**
************************************************************************/
void initialize_NP_solutions(struct Solution *solutions)
{
	int i, j;
	g_fes = 0;              /* ���ڿ��ܻ����ظ����ԣ��Է��������� 0 ֵ */

	boost::uniform_int<> real_x2((int)lower[2], (int)upper[2]);

	for (i = 0; i < NP; i++)
	{
		solutions[i].x[2] = real_x2(rng);
		boost::uniform_int<> real_x0((int)lower[0], (int)(upper[0] - 64 * (double(solutions[i].x[2]) / X2_SCALE))); // 10->5.0
		boost::uniform_int<> real_x1((int)lower[1], (int)(upper[1] - 128 * (double(solutions[i].x[2]) / X2_SCALE))); // 10->5.0
		solutions[i].x[0] = real_x0(rng);
		solutions[i].x[1] = real_x1(rng);
		object_function(solutions + i);
	}
}


/************************************************************************
* Function Name : obtain_fitness_max_min
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : ��� solution ��������������Сֵ�� ������ɨ��һ������
*
* Param : solutions: ָ�������NPά solution ����
* max: ָ�룬ָ��������ı������洢���ֵ
* min: ָ�룬ָ��������ı������洢��Сֵ
**
************************************************************************/
void obtain_fitness_max_min(struct Solution *solutions, double *max, double *min)
{
	int i;
	*max = solutions[0].fitness;
	*min = solutions[0].fitness;
	for (i = 1; i < NP; i++)
	{
		if (solutions[i].fitness < *min)
			*min = solutions[i].fitness;
		if (*max < solutions[i].fitness)
			*max = solutions[i].fitness;
	}
}


/************************************************************************
* Function Name : select_NICHING_SIZE
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : ��������� G ������һ����С, ʹ����boots ��uniform_int��
��������� rng ��ͷ�ļ� mulobjective_LAMCACO.h ����
*
* Param : G: ���飬����Ϊ2�� �洢���ܵ� niching size
* NICHING_SIZE: �洢ѡ�е�niching size
**
************************************************************************/
//void select_NICHING_SIZE(int *G, int *NICHING_SIZE)
//{
//	boost::uniform_int<> ui_01(0, 1);
//	*NICHING_SIZE = G[ui_01(rng)];
//}

int calculate_distance(int *x, int *y)
{
	double distance = 0;
	int    i;
	for (i = 0; i <  Dim; i++)
		distance += pow(x[i] - y[i], 2);
	return distance;
}


/************************************************************************
* Function Name : partition_into_crowds
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : �� crowds ���෽���� NP �� solutions �ֽ�� NP / NICHING_SIZE
�� niching�� ��ŵ� ��̬��ά���� niching_index ����ȥ
*
* Param : solutions: NP ����
* NICHING_SIZE: �洢ѡ�е�niching size
* niching_index: ��̬��ά���飬�洢�������
**
************************************************************************/
void partition_into_crowds(struct Solution *solutions, int niching_index[][NICHING_SIZE])
{
	vector <int> index;
	int i = 0, j, k, l, select_index, nearest_index;
	for (j = 0; j < NP; j++)
		index.push_back(j);
	while (!index.empty())
	{
		//cout << i << "size::" << NICHING_SIZE << endl;
		j = 0;
		/* ���ѡȡ index ������±� */
		boost::uniform_int<> ui(0, index.size() - 1);
		select_index = ui(rng);
		/* ��ѡ�е��±����õ� niching_index �� ���� index ��ɾ��*/
		niching_index[i][j] = index[select_index];
		index.erase(index.begin() + select_index);
		/* �齨 NICHING_SIZE ��С�� niching */
		for (j = 1; j < NICHING_SIZE; j++)
		{
			nearest_index = 0;
			/* �ҳ�ŷ����þ�������� */
			for (l = 1; l < index.size(); l++)
			if (calculate_distance(solutions[niching_index[i][0]].x, solutions[index[l]].x) <
				calculate_distance(solutions[niching_index[i][0]].x, solutions[index[nearest_index]].x))
					nearest_index = l;
			/* ���뵽���ڵ� niching ���� �ٴ� index ������ɾ��*/
			niching_index[i][j] = index[nearest_index];
			index.erase(index.begin() + nearest_index);
		}
		i++;
	}
}

/************************************************************************
* Function Name : complare
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : �ȽϺ������Ƚ�������Ĵ�С����С��������
*
* Param : a: solution
* b: solution
**
************************************************************************/
bool my_compare(struct Solution a, struct Solution b)
{
	return a.fitness < b.fitness;
}


void partition_into_speciation(struct Solution *solutions, int niching_index[][NICHING_SIZE])
{
	vector <int> index;
	int i = 0, j, k, l, select_index, nearest_index;

	sort(solutions, solutions + NP - 1, my_compare);
	for (j = 0; j < NP; j++)
		index.push_back(j);
	while (!index.empty())
	{
		j = 0;
		/* ѡ��õ� Ҳ���ǵ�һ��  */
		//boost::uniform_int<> ui(0, index.size() - 1);
		//select_index = ui(rng);
		select_index = 0;

		/* ��ѡ�е��±����õ� niching_index �� ���� index ��ɾ��*/
		niching_index[i][j] = index[select_index];
		index.erase(index.begin() + select_index);
		/* �齨 NICHING_SIZE ��С�� niching */
		for (j = 1; j < NICHING_SIZE; j++)
		{
			nearest_index = 0;
			/* �ҳ�ŷ����þ�������� */
			for (l = 1; l < index.size(); l++)
			if (calculate_distance(solutions[niching_index[i][0]].x, solutions[index[l]].x) <
				calculate_distance(solutions[niching_index[i][0]].x, solutions[index[nearest_index]].x))
				nearest_index = l;
			/* ���뵽���ڵ� niching ���� �ٴ� index ������ɾ��*/
			niching_index[i][j] = index[nearest_index];
			index.erase(index.begin() + nearest_index);
		}
		i++;
	}
}



/************************************************************************
* Function Name : calculate_weight
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : ����Ȩ��
************************************************************************/
void calculate_weight(struct Solution *s, double *s_weight, double sigma)
{
	int i;
	for (i = 0; i < NICHING_SIZE; i++)
		s_weight[i] = exp(-1 * pow(i, 2) / (2 * pow(sigma, 2) * pow(NICHING_SIZE, 2))) / (sigma * NICHING_SIZE * sqrt(2 * PI));
}


void calculate_probability(double *probability, double *s_weight)
{
	int i;
	double sum_of_weight = 0;
	for (i = 0; i < NICHING_SIZE; i++)
		sum_of_weight += s_weight[i];
	probability[0] = s_weight[0] / sum_of_weight;
	for (i = 1; i < NICHING_SIZE; i++)
		probability[i] = probability[i - 1] + s_weight[i] / sum_of_weight;
}


/************************************************************************
* Function Name : construct_NP_new_solutions
* Create Date : 2016/11/30
* Author/Corporation : bigzhao
**
Description : Ŀǰ�뷨�� ��̬���� NS �������飨���������Ͳ����� 
rank �ˣ� �������顢Ȩ������
*
* Param: solutions: �ɽ⼯��
* NICHING_SIZE: niching �Ĵ�С
* niching_index: �������±� ��ά����
* fs_max: ��� fitness
* fs_min: ��С fitness
* new_solutions: NP ���½�
**
************************************************************************/
void construct_NP_new_solutions(struct Solution *solutions, 
	int niching_index[][NICHING_SIZE], double fs_max, double fs_min, struct Solution *new_solutions)
{
	int    i, j, k, m;
	double fs_max_i, fs_min_i, sigma, prob, delta[Dim] = {0};
	struct Solution selected_solution, mu, niching_solutions[NICHING_SIZE];
	/* probability weight niching_solutions ��Ӧÿ�� niching �ĸ��� Ȩ�� �� */
	//probability = new double[NICHING_SIZE];
	//s_weight = new double[NICHING_SIZE];
	//niching_solutions = new struct Solution[NICHING_SIZE];
	double probability[NICHING_SIZE], s_weight[NICHING_SIZE], position[Dim];

	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		/* ��������һ��˳��� niching_solutions ��ֵ��*/
		fs_max_i = solutions[niching_index[i][0]].fitness;
		fs_min_i = solutions[niching_index[i][0]].fitness;
		niching_solutions[0] = solutions[niching_index[i][0]];
		for (j = 1; j < NICHING_SIZE; j++)
		{
			/* �ҳ�ÿ��С��������С fitness */
			if (fs_max_i < solutions[niching_index[i][j]].fitness)
				fs_max_i = solutions[niching_index[i][j]].fitness;
			if (solutions[niching_index[i][j]].fitness < fs_min_i)
				fs_min_i = solutions[niching_index[i][j]].fitness;
			niching_solutions[j] = solutions[niching_index[i][j]];
		}
		/*���� ����Ӧ���� �� 0.00000001 ��Ϊ�˱����ĸΪ0 */
		sigma = 0.1 + 0.3 * (exp((fs_max_i - fs_min_i) / (fs_max - fs_min + 0.0000000000001)));
		sort(niching_solutions, niching_solutions + NICHING_SIZE - 1, my_compare);
		/*for (j = 0; j < NICHING_SIZE; j++)
			cout << niching_solutions[j].fitness << " ";*/
		calculate_weight(niching_solutions, s_weight, sigma);
		/* ����ĸ����������ۼӵ� ���� 0.1 0.3 0.5 0.7 1 ��������Ϊ���������������һ��*/
		calculate_probability(probability, s_weight);
		/* ��ʼ������ ���� boots ���uniform_01���� ��mulobjective��������*/
		for (j = 0; j < NICHING_SIZE; j++)
		{
			prob = u01();
			for (k = 0; k < NICHING_SIZE; k++)
			{
				if (prob < probability[k])
					selected_solution = niching_solutions[k];
			}
			if (u01() < 0.5)
				mu = selected_solution;
			else
			{
				/* ע��߽����� */
				for (k = 0; k < Dim; k++)
				{
					mu.x[k] = selected_solution.x[k] + u01() * niching_solutions[0].x[k];
				}
				bounding_solution(&mu);
			}
			for (k = 0; k < Dim; k++)
			{
				for (m = 0; m < NICHING_SIZE; m++)
					delta[k] += abs(niching_solutions[m].x[k] - niching_solutions[j].x[k]);
				delta[k] *= u01() / (NICHING_SIZE - 1);
			}
			/* ���� dim ά �Ľ�*/
			for (k = 0; k < Dim; k++)
			{
				//if (u01() < 0.5)
				//{
					boost::normal_distribution<> nd(mu.x[k], delta[k]); // boots ��ĸ�˹�ֲ�����
					new_solutions[i * NICHING_SIZE + j].x[k] = nd(u01);
			/*	}
				else
				{
					boost::cauchy_distribution<> cd(mu.x[k], delta[k]);
					new_solutions[i * NICHING_SIZE + j].x[k] = cd(u01);
				}*/
				//cout << mu.x[k] << "  " << new_solutions[i * NICHING_SIZE + j].x[k] << endl;
				//cout << i * NICHING_SIZE + j << endl;
			}
			bounding_solution(new_solutions + i * NICHING_SIZE + j);
			object_function(new_solutions + i * NICHING_SIZE + j);
		}
	}
}


/************************************************************************
* Function Name : get_nearest_archive_solution
* Create Date : 2016/12/1
* Author/Corporation : bigzhao
**
Description : �����������Ŀ��ŷ����þ�������Ľ⣬�������±�
rank �ˣ� �������顢Ȩ������
*
* Param: solution: Ҫ�ȽϵĽ�
* source: ���Ƚϵ�Դ����
* src_length: ���еĳ���
* return: ����Ľ���±�
**
************************************************************************/
int get_nearest_archive_solution(struct Solution solution, struct Solution *source, int src_length)
{
	int nearest_index = 0, i;
	for (i = 1; i < src_length; i++)
	{
		if (calculate_distance(solution.x, source[i].x) < calculate_distance(solution.x, source[nearest_index].x))
			nearest_index = i;
	}
	return nearest_index;
}


/************************************************************************
* Function Name : local_searching
* Create Date : 2016/12/1
* Author/Corporation : bigzhao
**
Description : �ֲ�����
**
************************************************************************/
void local_searching(struct Solution *solutions, int *seeds, int niching_index[][NICHING_SIZE])
{
	/* fse_min �� seeds ���� fitness ��Сֵ  fse_max �����ֵ probability �Ǹ�������*/
	double fse_min, fse_max, probability[NP / NICHING_SIZE];  
	bool   flag = false;
	int    i, j, k;
	struct Solution temp;

	/* �������� seeds ����������Сֵ*/
	fse_max = solutions[niching_index[0][seeds[0]]].fitness;
	fse_min = solutions[niching_index[0][seeds[0]]].fitness;
	for (i = 1; i < NP / NICHING_SIZE; i++)
	{
		if (fse_max < solutions[niching_index[i][seeds[i]]].fitness)
			fse_max = solutions[niching_index[i][seeds[i]]].fitness;
		if (solutions[niching_index[i][seeds[i]]].fitness < fse_min)
			fse_min = solutions[niching_index[i][seeds[i]]].fitness;
	}

	if (fse_min <= 0)
	{
		fse_max += abs(fse_min) + 0.00000001;
		flag = true;
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		if (flag)
			probability[i] = (solutions[niching_index[i][seeds[i]]].fitness + abs(fse_min) + 0.00000001) /
			(fse_max + abs(fse_min) + 0.00000001);
		else
			probability[i] = solutions[niching_index[i][seeds[i]]].fitness / fse_max;
		if (u01() < probability[i])
		{
			for (k = 0; k < LOCAL_SEARCHING_TIMES; k++)
			{
				for (j = 0; j < Dim; j++)
				{
					boost::normal_distribution<> local_nd(solutions[niching_index[i][seeds[i]]].x[j], LOCAL_DELTA); // boots ��ĸ�˹�ֲ�����
					temp.x[j] = local_nd(u01);
				}
				bounding_solution(&temp);
				object_function(&temp);
				if (temp.fitness < solutions[niching_index[i][seeds[i]]].fitness)
					solutions[niching_index[i][seeds[i]]] = temp;
			}
		}
	}	


}



void LAMCACO(vector <struct Solution> *mul_object_found)
{
	struct Solution solutions[NP], new_solutions[NP], output_solutions[NP / NICHING_SIZE][NICHING_SIZE], result_seeds[NP / NICHING_SIZE];
	int    seeds[NP / NICHING_SIZE], seed;
	double fs_max, fs_min, radius;
	int    i, j, k;
	int    G[2] = { 10, 10 };
	//int    NICHING_SIZE = 10;  // ���� size �ȹ̶�Ϊ10 ԭ������G�������ѡ
	int    nearest_archive_solution_index;
	int    niching_index[NP / NICHING_SIZE][NICHING_SIZE];

	initialize_NP_solutions(solutions);
	while (g_fes < MAX_FES)
	{
		obtain_fitness_max_min(solutions, &fs_max, &fs_min);
		partition_into_crowds(solutions, niching_index);
		construct_NP_new_solutions(solutions, niching_index, fs_max, fs_min, new_solutions);

		for (i = 0; i < NP; i++)
		{
			nearest_archive_solution_index = get_nearest_archive_solution(new_solutions[i], solutions, NP);    /* ������ solutions ���±� */
			if (new_solutions[i].fitness < solutions[nearest_archive_solution_index].fitness)
				solutions[nearest_archive_solution_index] = new_solutions[i];
		}
		for (i = 0; i < NP / NICHING_SIZE; i++)
		{
			seed = 0;
			for (j = 1; j < NICHING_SIZE; j++)
			{
				if (solutions[niching_index[i][j]].fitness < solutions[niching_index[i][seed]].fitness)
					seed = j;
			}
			seeds[i] = seed;
		}

		local_searching(solutions, seeds, niching_index);     /* һ�����ʽ��оֲ����� */
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		for (j = 0; j < NICHING_SIZE; j++)
		{
			output_solutions[i][j] = solutions[niching_index[i][j]];
		}
		sort(&output_solutions[i][0], &output_solutions[i][NICHING_SIZE], my_compare);
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
		result_seeds[i] = output_solutions[i][0];
	sort(result_seeds, result_seeds + NP / NICHING_SIZE, my_compare);
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		if (result_seeds[i].fitness < THRESHOLD)
			{
				for (k = 0; k < mul_object_found->size(); k++)
				{
					radius = (*mul_object_found)[k].x[2] * 6.4 * radius_factor;
					if (abs((*mul_object_found)[k].x[0] - result_seeds[i].x[0]) < radius)
						break;
				}
				if (k == mul_object_found->size())
					mul_object_found->push_back(result_seeds[i]);
			}
	}
}


void LAMSACO(vector <struct Solution> *mul_object_found)
{
	struct Solution solutions[NP], new_solutions[NP], output_solutions[NP / NICHING_SIZE][NICHING_SIZE], 
		result_seeds[NP / NICHING_SIZE];
	int    seeds[NP / NICHING_SIZE], seed;
	double fs_max, fs_min, radius;
	int    i, j, k;
	int    G[2] = { 10, 10 };
	//int    NICHING_SIZE = 10;  // ���� size �ȹ̶�Ϊ10 ԭ������G�������ѡ
	int    nearest_archive_solution_index;
	int    niching_index[NP / NICHING_SIZE][NICHING_SIZE];

	initialize_NP_solutions(solutions);
	while (g_fes < MAX_FES)
	{
		obtain_fitness_max_min(solutions, &fs_max, &fs_min);
		partition_into_speciation(solutions, niching_index);
		construct_NP_new_solutions(solutions, niching_index, fs_max, fs_min, new_solutions);

		for (i = 0; i < NP; i++)
		{
			/* ������ solutions ���±� */
			nearest_archive_solution_index = get_nearest_archive_solution(new_solutions[i], solutions, NP);    
			if (new_solutions[i].fitness < solutions[nearest_archive_solution_index].fitness)
				solutions[nearest_archive_solution_index] = new_solutions[i];
		}
		for (i = 0; i < NP / NICHING_SIZE; i++)
		{
			seed = 0;
			for (j = 1; j < NICHING_SIZE; j++)
			{
				if (solutions[niching_index[i][j]].fitness < solutions[niching_index[i][seed]].fitness)
					seed = j;
			}
			seeds[i] = seed;
		}

		local_searching(solutions, seeds, niching_index);     /* һ�����ʽ��оֲ����� */
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		for (j = 0; j < NICHING_SIZE; j++)
		{
			output_solutions[i][j] = solutions[niching_index[i][j]];
		}
		sort(&output_solutions[i][0], &output_solutions[i][NICHING_SIZE], my_compare);
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
		result_seeds[i] = output_solutions[i][0];
	sort(result_seeds, result_seeds + NP / NICHING_SIZE, my_compare);
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		if (result_seeds[i].fitness < THRESHOLD)
		{
			for (k = 0; k < mul_object_found->size(); k++)
			{
				radius = (*mul_object_found)[k].x[2] * 6.4 * radius_factor;
				if (abs((*mul_object_found)[k].x[0] - result_seeds[i].x[0]) < radius)
					break;
			}
			if (k == mul_object_found->size())
				mul_object_found->push_back(result_seeds[i]);
		}
	}
}


/************************************************************************
* Function Name : get_stddev
* Create Date : 2016/11/26
* Author/Corporation : bigzhao
**
Description : �����׼��
*
* Param : x: �����������
* mean: ƽ��ֵ
* length: ���鳤��
**
************************************************************************/
double get_stddev(double *x, double mean, int length)
{
	int    i;
	double sum = 0.0;

	for (i = 0; i < length; i++)
		sum += pow(x[i] - mean, 2);

	return sqrt(sum / length);
}
/************************************************************************
* Function Name : generate_new_population
* Create Date : 2016/11/26
* Author/Corporation : bigzhao
**
Description : EDAs ������Ҫ��һ�����ڣ�ͨ����˹�ֲ�������һ����Ⱥ������ĸ�
˹����ʹ�� boots ��� normal_distribution���÷��ǰٶȲ�ģ��պ�úú���ĥһ
��
*
* Param : pop: ����Ⱥ
* new_pop: ����Ⱥ
**
************************************************************************/
void generate_new_solutions_use_EDA(struct Solution *solutions, 
	int niching_index[][NICHING_SIZE], struct Solution *new_solutions)
{
	int    i, j, k, m = 0;
	double mean, stddev, sum = 0;
	double temp_dimension[NICHING_SIZE];

	for (k = 0; k < NP / NICHING_SIZE; k++)
	{
		for (i = 0; i < Dim; i++)
		{
			sum = 0;
			for (j = 0; j < NICHING_SIZE; j++)
			{
				temp_dimension[j] = solutions[niching_index[k][j]].x[i];
				sum += temp_dimension[j];
			}

			mean = sum / NICHING_SIZE;
			stddev = get_stddev(temp_dimension, mean, NICHING_SIZE);
			boost::normal_distribution<> nd(mean, stddev); // boots ��ĸ�˹�ֲ�����

			for (j = 0; j < NICHING_SIZE; j++)
			{
				new_solutions[k * NICHING_SIZE + j].x[i] = nd(u01);   // ��˹�ֲ������µ�ֵ����ɸ���
			}

		}
	}
	// ���� fitness
	for (i = 0; i < NP; i++)
	{
		bounding_solution(new_solutions + i);
		object_function(new_solutions + i);
	}
}

int get_nearest_archive_solution_niche(struct Solution solution, struct Solution *source, int *index, int src_length)
{
	int i, nearest_index = 0;
	for (i = 1; i < src_length; i++)
	{
		if (calculate_distance(solution.x, source[index[i]].x) < calculate_distance(solution.x, source[index[nearest_index]].x))
			nearest_index = i;
	}
	return index[nearest_index];
}

/* Local Search-Based MCEDA */
void LMCEDA(vector <struct Solution> *mul_object_found)
{
	struct Solution solutions[NP], new_solutions[NP], output_solutions[NP / NICHING_SIZE][NICHING_SIZE],
		result_seeds[NP / NICHING_SIZE];
	int    seeds[NP / NICHING_SIZE], seed;
	double fs_max, fs_min, radius;
	int    i, j, k;
	int    G[2] = { 10, 10 };
	//int    NICHING_SIZE = 10;  // ���� size �ȹ̶�Ϊ10 ԭ������G�������ѡ
	int    nearest_archive_solution_index;
	int    niching_index[NP / NICHING_SIZE][NICHING_SIZE];

	initialize_NP_solutions(solutions);
	while (g_fes < MAX_FES)
	{
		//obtain_fitness_max_min(solutions, &fs_max, &fs_min);
		partition_into_crowds(solutions, niching_index);
		generate_new_solutions_use_EDA(solutions, niching_index, new_solutions);

		//for (i = 0; i < NP / NICHING_SIZE; i++)
		//{
		//	for (j = 0; j < NICHING_SIZE; j++)
		//	{
		//		/* ������ solutions ���±� */
		//		nearest_archive_solution_index = get_nearest_archive_solution_niche(new_solutions[i], solutions, niching_index[j], NICHING_SIZE);
		//		if (new_solutions[i * NICHING_SIZE + j].fitness < solutions[nearest_archive_solution_index].fitness)
		//			solutions[nearest_archive_solution_index] = new_solutions[i];
		//	}
		//}

		for (i = 0; i < NP; i++)
		{
				/* ������ solutions ���±� */
				nearest_archive_solution_index = get_nearest_archive_solution(new_solutions[i], solutions, NP);
				if (new_solutions[i].fitness < solutions[nearest_archive_solution_index].fitness)
					solutions[nearest_archive_solution_index] = new_solutions[i];
		}
		for (i = 0; i < NP / NICHING_SIZE; i++)
		{
			seed = 0;
			for (j = 1; j < NICHING_SIZE; j++)
			{
				if (solutions[niching_index[i][j]].fitness < solutions[niching_index[i][seed]].fitness)
					seed = j;
			}
			seeds[i] = seed;
		}

		local_searching(solutions, seeds, niching_index);     /* һ�����ʽ��оֲ����� */
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		for (j = 0; j < NICHING_SIZE; j++)
		{
			output_solutions[i][j] = solutions[niching_index[i][j]];
		}
		sort(&output_solutions[i][0], &output_solutions[i][NICHING_SIZE], my_compare);
	}
	for (i = 0; i < NP / NICHING_SIZE; i++)
		result_seeds[i] = output_solutions[i][0];
	sort(result_seeds, result_seeds + NP / NICHING_SIZE, my_compare);
	for (i = 0; i < NP / NICHING_SIZE; i++)
	{
		if (result_seeds[i].fitness < THRESHOLD)
		{
			for (k = 0; k < mul_object_found->size(); k++)
			{
				radius = (*mul_object_found)[k].x[2] * 6.4 * radius_factor;
				if (abs((*mul_object_found)[k].x[0] - result_seeds[i].x[0]) < radius)
					break;
			}
			if (k == mul_object_found->size())
				mul_object_found->push_back(result_seeds[i]);
		}
	}
}



int main(int argc, char** argv)
{
	int i, m;
	srand(time(NULL));
	FILE* f = 0;
	FILE* outf;
	double scale;
	//struct Solution solutions[NP];
	vector <struct Solution> mul_object_found;
	char window_name[100] = "DE���1";
	HOGDescriptor hog2;
	hog2.setSVMDetector(hog2.getDefaultPeopleDetector());
	FILE *fp;
	if ((fp = fopen("lmceda_nsde.txt", "w")) == NULL) {
		printf("File cannot be opened/n");
		exit(1);
	}
	//���ļ��ж�ȡͼ��  
	for (m = START; m < ITER; m++)
	{
		IplImage *pImage = cvLoadImage(g_filename[m], CV_LOAD_IMAGE_UNCHANGED);
		for (int tri = 0; tri < trials; tri++)
		{
			mul_object_found.clear();
			cout << g_filename[m] << endl;
			readImg(g_filename[m]);

			int levels = upper[2] - lower[2];

			for (i = 0; i < 800; i++)
			for (int j = 0; j < 800; j++)
			for (int k = 0; k < levels + 1; k++)
				weight[i][j][k] = -10000;
			memset(block_des, 0, sizeof(block_des));
			mul_object_found.clear();
			clock_t start, end;
			start = clock();
			scale_the_pics();


			/* �����￪ʼ�� LAMCACO �Ĵ���--------��ע�� 2016/12/2 22��44------------------------------------- */
			//LAMCACO(&mul_object_found);


			//end = clock();
			//double cal_time = double(end - start) / CLOCKS_PER_SEC;
			//cout << "LAMCACO �ķ�ʱ�䣨�룩:" << cal_time << endl;
			////cout << "Repeat times:" << count_repeated << endl;
			////cout << "Repeat block times:" << count_re_block << endl;
			////cout << "Block times:" << count_block << endl;


			//CvRect rect;
			//scale = winsize*stepz + lower[2];
			//scale = scale / X2_SCALE;    // 10->5.0
			//int sx = cvCeil(stepx / scale);

			//cout << "found size:" << mul_object_found.size() << endl;

			///* �ҵĲ��Դ��� */
			//for (i = 0; i < mul_object_found.size(); i++)
			//{
			//	rect.x = mul_object_found[i].x[0];
			//	rect.y = mul_object_found[i].x[1];

			//	rect.width = 64 * mul_object_found[i].x[2] / X2_SCALE; // 10->5.0
			//	rect.height = 128 * mul_object_found[i].x[2] / X2_SCALE;      //try the bin de // 10->5.0

			//	if (size_factor > 0 && size_factor < 1)
			//	{                               //to resize the rect
			//		rect.x += rect.width * (1 - size_factor) / 2;
			//		rect.y += rect.height * (1 - size_factor) / 2;
			//		rect.width = rect.width * size_factor;
			//		rect.height = rect.height * size_factor;
			//	}
			//	cv::rectangle(img_for_NSDE, rect, cv::Scalar(255, 255, 255), 2);
			//}
			/*LAMCACO �������-----------------------------------------------------------------------------*/


			/* �����￪ʼ�� LAMSACO �Ĵ��� ��ע�� 2016/12/2 22��44-------------------------------------------*/
			//start = clock();
			//LAMSACO(&mul_object_found);
			//end = clock();
			//cal_time = double(end - start) / CLOCKS_PER_SEC;
			//cout << "LAMCACO �ķ�ʱ�䣨�룩:" << cal_time << endl;
			////cout << "Repeat times:" << count_repeated << endl;
			////cout << "Repeat block times:" << count_re_block << endl;
			////cout << "Block times:" << count_block << endl;


			//cout << "found size:" << mul_object_found.size() << endl;

			///* �ҵĲ��Դ��� */
			//for (i = 0; i < mul_object_found.size(); i++)
			//{
			//	rect.x = mul_object_found[i].x[0];
			//	rect.y = mul_object_found[i].x[1];

			//	rect.width = 64 * mul_object_found[i].x[2] / X2_SCALE; // 10->5.0
			//	rect.height = 128 * mul_object_found[i].x[2] / X2_SCALE;      //try the bin de // 10->5.0


			//	if (size_factor > 0 && size_factor < 1)
			//	{                               //to resize the rect
			//		rect.x += rect.width * (1 - size_factor) / 2;
			//		rect.y += rect.height * (1 - size_factor) / 2;
			//		rect.width = rect.width * size_factor;
			//		rect.height = rect.height * size_factor;
			//	}
			//	cv::rectangle(img_for_LAMSACO, rect, cv::Scalar(255, 255, 255), 2);
			//}
			/* LAMSACO �������-------------------------------------------------------------------------*/

			/* ���Դ������ */

			//for (i = 0; i<multi_found; i++){


			//	rect.x = best[i].x[0];
			//	rect.y = best[i].x[1];

			//	rect.width = 64 * best[i].x[2] / 10;
			//	rect.height = 128 * best[i].x[2] / 10;      //try the bin de


			//	if (size_factor>0 && size_factor<1){                               //to resize the rect
			//		rect.x += rect.width*(1 - size_factor) / 2;
			//		rect.y += rect.height*(1 - size_factor) / 2;
			//		rect.width = rect.width*size_factor;
			//		rect.height = rect.height*size_factor;
			//	}


			//	cout << window_name << ": " << endl << "fitness:" << best[i].fitness << endl;
			//	cout << "x,y,winsize" << rect.x << endl << rect.y << endl << best[i].x[2] << endl;

			//	window_name[6]++;


			//cv::rectangle(img_for_NSDE, rect, cv::Scalar(255, 255, 255), 2);

			/* �����￪ʼ�� LMCEDA �Ĵ���------------------------------------------- */
			LMCEDA(&mul_object_found);


			end = clock();
			double cal_time = double(end - start) / CLOCKS_PER_SEC;
			cout << "LMCEDA �ķ�ʱ�䣨�룩:" << cal_time << endl;
			fprintf(fp, "%lf\n", cal_time);

			//cout << "Repeat times:" << count_repeated << endl;
			//cout << "Repeat block times:" << count_re_block << endl;
			//cout << "Block times:" << count_block << endl;

			// ע�� 2016.12.14
			CvRect rect;
			scale = winsize*stepz + lower[2];
			scale = scale / X2_SCALE;    // 10->5.0
			int sx = cvCeil(stepx / scale);

			cout << "found size:" << mul_object_found.size() << endl;

			/* �ҵĲ��Դ��� */
			for (i = 0; i < mul_object_found.size(); i++)
			{
				rect.x = mul_object_found[i].x[0];
				rect.y = mul_object_found[i].x[1];

				rect.width = 64 * mul_object_found[i].x[2] / X2_SCALE; // 10->5.0
				rect.height = 128 * mul_object_found[i].x[2] / X2_SCALE;      //try the bin de // 10->5.0

				if (size_factor > 0 && size_factor < 1)
				{                               //to resize the rect
					rect.x += rect.width * (1 - size_factor) / 2;
					rect.y += rect.height * (1 - size_factor) / 2;
					rect.width = rect.width * size_factor;
					rect.height = rect.height * size_factor;
				}
				cv::rectangle(img_for_LMCEDA, rect, cv::Scalar(255, 0, 0), 4);
			}
			//ע�� 2016.12.14
			/*LAMCACO �������-----------------------------------------------------------------------------*/



			//}
			/* ��ʼ MultiScale �㷨---------------------------------------------------------------------------*/
			//start = clock();
			//std::vector<cv::Rect> regions;   //for multiscale detection

			//hog2.detectMultiScale(img_for_multiscale, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

			//for (size_t b = 0; b < regions.size(); b++)
			//{
			//	cv::rectangle(img_for_multiscale, regions[b], cv::Scalar(0, 0, 255), 2);
			//}                             //for multiscale detection
			//end = clock();
			//double time0 = double(end - start) / CLOCKS_PER_SEC;
			//cout << "multiscale �ķ�ʱ�䣨�룩:" << time0 << endl;
			///* MultiScale �㷨���� ����------------------------------------------------------------------------*/


			////cout << "The lower and upper is:" << lower[2] << "     " << upper[2];
			///* ��ʾ��� */
			//cv::imshow("Multiscale���", img_for_multiscale);
			cv::imshow("LMCEDA", img_for_LMCEDA);
			//cv::imshow("LAMSACO", img_for_LAMSACO);
			cvWaitKey();

		}
		//system("pause");
	}
}


