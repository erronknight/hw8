
// Used from Joseph Straceski's code
// https://github.com/jstraceski/cs5335hw-gazebo/tree/hw07

// modified sarah coffen nov 2020

#include <iostream>
#include <thread>
#include <math.h>
#include <vector>
#include <deque>
#include <chrono>
#include <cmath>
#include <time.h>
#include <mutex>

#include "viz.hh"
#include "robot.hh"
#include "cam.hh"

/* Surface to store current scribbles */
std::mutex mx2;

using std::cout;
using std::endl;

#define PI 3.14159265

std::vector<std::vector<bool>> hit_lookup;
std::vector<float> coll = {0, 0, 0};
std::vector<float> drive_shift = {0, 0, 0};
std::vector<float> save_guess = {0, 0, 0};
std::vector<float> guess = {0, 0, 0};
std::vector<float> offset = {0, 0, 0};
std::vector<float> last_guess = {0, 0, 0};
std::vector<float> thread_pos = {0, 0, 0};

std::deque<std::vector<int>> target_path;
std::deque<std::vector<int>> target_path_thread;
std::vector<int> target_point = {};

bool has_path = false;
bool hit_drive = false;
bool coll_delete = false;
bool left_follow = false;
bool right_follow = false;
bool searching = false;
bool at_goal = false;
bool new_goal = false;

// time variables
// float time_delt = 0;
// float time_delta = 0;

time_t time_delt = 0;
time_t time_delta = 0;

// tuned values
float grid_div = 0.2f;
float input_scalar = 1.05f;
float predict_weight = 20.0f;
float hit_weight = 0.1f;
int grid_offset = 28;
int setup_c = 0;
int check_count2 = 0;
int check_limit2 = 50;

// robot wheel values
float slip_constant = 2.0f * 0.68f;

float chassis_dy = 0.216f;
float wheel_radius = 0.050f;
float wheel_width = 0.040f;
float wheel_y0 = (chassis_dy * 0.5f + wheel_width * 0.6f + 0.010f);
float wheel_tread = wheel_y0 * 2;

// float last_time = 0;
time_t last_time = 0;

std::vector<float> last_drive = {0, 0};
std::vector<float> guess_diff = {0, 0};

int size = std::floor((float(grid_offset) * 2.0) / grid_div);
float radius = float(grid_offset);

std::vector<std::vector<float>> tile_grid(size, std::vector<float>(size, -1));
std::vector<std::vector<float>> tile_grid_thread(size, std::vector<float>(size, -1));

float cam_left = 0;
float cam_mid = 0;
float cam_right = 0;

double get_30_deg(double va) {
    return (2* (va / sqrt(3.0)));
}

// theta <= 40
double get_deg_adj(double va, double thet) {
    return (va / cos((thet / 180) * PI));
}

double mod_val(double aa) {
    return 1.3 * get_deg_adj(2.0 *(aa) / 300.0, (400 - (aa) / 10));
}

std::vector<std::vector<int>> get_im_vals(cv::Mat frame) {
    // sky ~(178, 178, 178)
    // floor ~(155, 155, 155)
    // walls avg < 120
    std::vector<std::vector<int>> v;
    std::vector<int> vmid = {0, 0, 0};
    std::vector<int> vlft = {0, 0, 0};
    std::vector<int> vrgt = {0, 0, 0};

    v = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
        };

    cv::Vec3b pc;
    if (frame.cols > 0 && frame.cols == 800 && frame.rows == 800) {
        int cmid = 400;
        int clft = 100;
        int crgt = 700;
        int avg = 0;
        bool m_sky = true;
        bool l_sky = true;
        bool r_sky = true;

        for (int rr = 0; rr < frame.rows; rr+=3) {
            // left
            pc = frame.at<cv::Vec3b>(rr,clft);
            avg = ((int)pc.val[0] + (int)pc.val[1] + (int)pc.val[2]) / 3;

            if (avg < 120 && l_sky) {
                l_sky = false;
                vlft[0] = rr;
            } else if (!l_sky) {
                vlft[1] = rr - vlft[0];
                vlft[2] = 799 - rr;
            }

            // mid
            pc = frame.at<cv::Vec3b>(rr,cmid);
            avg = ((int)pc.val[0] + (int)pc.val[1] + (int)pc.val[2]) / 3;
            if (avg < 120 && m_sky) {
                m_sky = false;
                vmid[0] = rr;
            } else if (!m_sky) {
                vmid[1] = rr - vmid[0];
                vmid[2] = 799 - rr;
            }

            // right
            pc = frame.at<cv::Vec3b>(rr,crgt);
            avg = ((int)pc.val[0] + (int)pc.val[1] + (int)pc.val[2]) / 3;
            if (avg < 120 && r_sky) {
                r_sky = false;
                vrgt[0] = rr;
            } else if (!r_sky) {
                vrgt[1] = rr - vrgt[0];
                vrgt[2] = 799 - rr;
            }
        }
        
        // cout << "mid framasmefmw asd " << (int)pixelColor.val[0] << " " << (int)pixelColor.val[1] << " " << (int)pixelColor.val[2] << endl;
        v[0][0] = vlft[0];
        v[0][1] = vlft[1];
        v[0][2] = vlft[2];

        v[1][0] = vmid[0];
        v[1][1] = vmid[1];
        v[1][2] = vmid[2];

        v[2][0] = vrgt[0];
        v[2][1] = vrgt[1];
        v[2][2] = vrgt[2];
    }
    // basically frame not ready
    return v;
}

void print_arr(std::vector<float> v)
{
    for (auto elem : v)
    {
        cout << elem << " ";
    }
    cout << endl;
}

void draw_point(float x, float y, float r, float g, float b)
{

    float x_delt = radius - x;
    float y_delt = radius - y;

    float ang = atan2(y_delt, x_delt);
    float dist = sqrt((y_delt * y_delt) + (x_delt * x_delt));

    viz_hit(-dist / float(radius) * 1.9f, ang, r, g, b);
}

void add_point(float x, float y, float r, float g, float b, std::vector<std::vector<float>> *point_list)
{
    float min = FLT_MAX;

    for (const std::vector<float> &master : *point_list)
    {
        float x_dlt = (x - master[0]);
        float y_dlt = (y - master[1]);

        float dist = x_dlt * x_dlt + y_dlt * y_dlt;

        if (dist < min)
        {
            min = dist;
        }
    }

    if (min > 0.5f)
    {
        std::vector<float> out = {x, y};
        point_list->emplace_back(out);
        draw_point(x, y, r, g, b);
    }
}

void draw_path(std::deque<std::vector<int>> d_path)
{
    for (int i = 0; i < d_path.size(); ++i)
    {
        std::vector<int> point = d_path[i];
        draw_point(float(point[0] * grid_div), float(point[1] * grid_div), 0.0f, 1.0f, 1.0f);
    }
}

void cmd_vel(Robot *robot, float vx, float vy, bool print)
{
    robot->set_vel(vx, vy);
}

// apply a motion to the robot and predict the outcome
void cmd_vel2(Robot *robot, float vx, float vy, bool print)
{
    float tot = guess_diff[0] * guess_diff[0] + guess_diff[1] * guess_diff[1];
    float coll_tot = coll[0] * coll[0] + coll[1] * coll[1];

    float pred_vec = guess_diff[0] * coll[0] + coll[1] * guess_diff[1];
    float diff_scal = ((guess_diff[0] - coll[0]) * (guess_diff[0] - coll[0])) + ((guess_diff[1] - coll[1]) * (guess_diff[1] - coll[1]));

    float d_s = 0.2 * clamp(1, diff_scal / 0.01f, 3);

    if (coll_delete && ((abs(pred_vec) > 0 && coll_tot > 0) || abs(coll[2] - guess_diff[2]) > 0.001))
    {
        float a1 = atan2(coll[1], coll[0]);
        float a2 = atan2(guess_diff[1], guess_diff[0]);
        float tang = (a1 - a2);

        if (guess_diff[2] > PI)
        {
            guess_diff[2] -= PI * 2;
        }
        else if (guess_diff[2] < -PI)
        {
            guess_diff[2] += PI * 2;
        }

        if (tang > PI)
        {
            tang -= PI * 2;
        }
        else if (tang < -PI)
        {
            tang += PI * 2;
        }

        cout << "@@@@@@@@@@@@@@ " << tang << " " << drive_shift[0] << " " << drive_shift[1] << endl;
        if ((tang > 0.1 && abs(last_drive[0] + last_drive[1]) / 2.0f > 0.1f) || coll[2] < guess_diff[2])
        {
            drive_shift[1] += d_s;
            drive_shift[0] -= d_s;
        }

        if ((tang < -0.1 && abs(last_drive[0] + last_drive[1]) / 2.0f > 0.1f) || coll[2] > guess_diff[2])
        {
            drive_shift[1] -= d_s;
            drive_shift[0] += d_s;
        }

        cout << guess_diff[0] << " " << guess_diff[1] << endl;
        cout << coll[0] << " " << coll[1] << endl;

        last_drive = {0, 0};
        cout << "@@@@@@@@@@@@@@ " << tang << " " << drive_shift[0] << " " << drive_shift[1] << endl;

        if (tot < coll_tot - 0.001 || pred_vec < 0)
        {
            drive_shift[0] -= d_s;
            drive_shift[1] -= d_s;
        }
        else if (tot > coll_tot + 0.001)
        {
            drive_shift[0] += d_s;
            drive_shift[1] += d_s;
        }
        cout << "@@@@@@@@@@@@@@ " << tang << " " << drive_shift[0] << " " << drive_shift[1] << endl;
    }

    last_drive = {last_drive[0] + vx, last_drive[1] + vy};

    cout << "vx:" << vx << " vy:" << vy << endl;
    cout << "tot:" << tot << " ct:" << coll_tot << " t:" << coll_delete << " d" << abs(time_delta) << " l:" << pred_vec << " " << guess_diff[2] << " as:" << coll[2] << endl;

    if (coll_delete)
    {
        coll_delete = false;
        coll[0] = 0;
        coll[1] = 0;
        coll[2] = 0;
    }

    if (abs(time_delta) < 1000)
    {
        coll[0] += ((vx + vy) / 2.0f * time_delta * wheel_radius * PI * slip_constant * cos(guess[2]));
        coll[1] += ((vx + vy) / 2.0f * time_delta * wheel_radius * PI * slip_constant * sin(guess[2]));
        coll[2] += -((vx - vy) / 2.0f * time_delta * wheel_radius * PI * slip_constant / (wheel_y0 * PI * 2));
    }

    drive_shift[0] = clamp((-10.0f * 0.2f) / input_scalar, drive_shift[0], (10.0f * 0.2f) / input_scalar);
    drive_shift[1] = clamp((-10.0f * 0.2f) / input_scalar, drive_shift[1], (10.0f * 0.2f) / input_scalar);

    robot->set_vel(vx / input_scalar - drive_shift[0], vy / input_scalar - drive_shift[1]);
}

// apply a value shift in a plus pattern arround the given x, y position
void apply_adj(int x, int y, float v, std::vector<std::vector<float>> &grid)
{
    int ax = x + 1;
    int ay = y + 0;

    int bx = x - 1;
    int by = y + 0;

    int cx = x + 0;
    int cy = y + 1;

    int dx = x + 0;
    int dy = y - 1;

    if (ax >= 0 && ay >= 0 && ax < grid.size() && ay < grid[0].size())
    {
        if (grid[ax][ay] == -1)
        {
            grid[ax][ay] = 0;
        }
        grid[ax][ay] += v;
        grid[ax][ay] = clamp(-0.99, grid[ax][ay], 2.0);
    }

    if (bx >= 0 && by >= 0 && bx < grid.size() && by < grid[0].size())
    {
        if (grid[bx][by] == -1)
        {
            grid[bx][by] = 0;
        }
        grid[bx][by] += v;
        grid[bx][by] = clamp(-0.99, grid[bx][by], 2.0);
    }

    if (cx >= 0 && cy >= 0 && cx < grid.size() && cy < grid[0].size())
    {
        if (grid[cx][cy] == -1)
        {
            grid[cx][cy] = 0;
        }
        grid[cx][cy] += v;
        grid[cx][cy] = clamp(-0.99, grid[cx][cy], 2.0);
    }

    if (dx >= 0 && dy >= 0 && dx < grid.size() && dy < grid[0].size())
    {
        if (grid[dx][dy] == -1)
        {
            grid[dx][dy] = 0;
        }
        grid[dx][dy] += v;
        grid[dx][dy] = clamp(-0.99, grid[dx][dy], 2.0);
    }
}

// is the tile adjacent to a wall
bool is_adj(int x, int y, float v, std::vector<std::vector<float>> &grid, int d)
{

    int width = grid.size();
    int height = grid[0].size();

    for (int xs = -d; xs <= d; ++xs)
    {
        for (int ys = -d; ys <= d; ++ys)
        {
            int tx = x + xs;
            int ty = y + ys;
            if (tx >= 0 && ty >= 0 && tx < width && ty < height)
            {
                if (tile_grid[tx][ty] > v)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

// update the costmap
bool plot_point_b(int x, int y, bool add, std::vector<std::vector<float>> &grid, float draw)
{

    if (x < 0 || y < 0 || x >= grid.size() || y >= grid[0].size())
    {
        cout << x << " " << y << endl;
        return true;
    }

    if (grid[x][y] == -1)
    {
        grid[x][y] = 0;
    }

    if (grid[x][y] < -0.9)
    {
        grid[x][y] = -0.9;
    }

    if (add)
    {
        grid[x][y] += 0.45f;
        apply_adj(x, y, 0.1, grid);
    }
    else
    {
        grid[x][y] -= 0.45f;
        apply_adj(x, y, -0.1, grid);
        ;
    }

    grid[x][y] = clamp(-0.99, grid[x][y], 2.0);

    if (draw)
    {
        draw_point((x * grid_div), (y * grid_div), clamp(0, grid[x][y], 1.0f) / 1.0f, 0, 0);
    }
    return false;
}

// check the value fo the given tile and the adjacent tiles if denoted
bool check_val(int x, int y, float val, bool adj, std::vector<std::vector<float>> &grid, int range)
{
    if (x < 0 || y < 0 || x >= grid.size() || y >= grid[0].size())
    {
        return true;
    }

    if (grid[x][y] > val || (adj && is_adj(x, y, val, grid, range)))
    {
        return true;
    }

    return false;
}

// code modified from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
bool bresenham_int(int x1, int y1, int x2, int y2, bool hit, bool find, std::vector<std::vector<float>> &grid, bool draw, int range)
{
    int delta_x(x2 - x1);
    // if x1 == x2, then it does not matter what we set here
    signed char const ix((delta_x > 0) - (delta_x < 0));
    delta_x = std::abs(delta_x) << 1;

    int delta_y(y2 - y1);
    // if y1 == y2, then it does not matter what we set here
    signed char const iy((delta_y > 0) - (delta_y < 0));
    delta_y = std::abs(delta_y) << 1;

    if (!find && plot_point_b(x1, y1, false, grid, draw))
    {
        return false;
    }

    if (find && check_val(x1, y1, 0, false, grid, 0))
    {
        cout << "o1:" << x1 << " " << y1 << " " << tile_grid[x1][y1] << endl;
        return false;
    }

    bool init = false;
    int last_x = 0;
    int last_y = 0;

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            if (find && init && check_val(last_x, last_y, 0, true, grid, range))
            {
                cout << "l1:" << last_x << " " << last_y << endl;
                return false;
            }

            if (!find && init && plot_point_b(last_x, last_y, false, grid, draw))
            {
                return false;
            }

            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            last_x = x1;
            last_y = y1;

            init = true;
        }
    }
    else
    {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            if (find && init && check_val(last_x, last_y, 0, true, grid, range))
            {
                cout << "l2:" << last_x << " " << last_y << endl;
                return false;
            }

            if (!find && init && plot_point_b(last_x, last_y, false, grid, draw))
            {
                return false;
            }
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            error += delta_x;
            y1 += iy;

            last_x = x1;
            last_y = y1;

            init = true;
        }
    }

    if (last_x < 0 || last_y < 0 || last_x >= size || last_y >= size)
    {

        cout << "l3:" << last_x << " " << last_y << endl;
        return false;
    }

    if (!find)
    {
        if (hit)
        {
            plot_point_b(last_x, last_y, true, grid, draw);
        }
        else if (tile_grid[last_x][last_y] < 0)
        {
            plot_point_b(last_x, last_y, false, grid, draw);
        }
    }

    if (find && init && check_val(last_x, last_y, 0, false, grid, 5))
    {
        cout << "l4:" << last_x << " " << last_y << endl;
        return false;
    }

    return true;
}

// plot a path from global positions [[xf1, yf1], [xf2, yf2]]
//  hit makes the algorithm short circuit
//  find will short circuit when the wall threshold has been tripped and will return false if the line is interrupted
//  grid to populate and read from
//  draw the line to the screen
//  the range of the trip threshold, the range is the 1/s of a cube around each search point
bool bresenham(float xf1, float yf1, float xf2, float yf2, bool hit, bool find, std::vector<std::vector<float>> &grid, bool draw, int range)
{
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    if (!find)
    {
        x1 = std::floor((grid_offset + xf1) / grid_div);
        y1 = std::floor((grid_offset + yf1) / grid_div);
        x2 = std::floor((grid_offset + xf2) / grid_div);
        y2 = std::floor((grid_offset + yf2) / grid_div);
    }
    else
    {
        x1 = xf1;
        y1 = yf1;
        x2 = xf2;
        y2 = yf2;
    }

    return bresenham_int(x1, y1, x2, y2, hit, find, grid, draw, range);
}

// reduce the unnecesary points in an A* path using the grid
std::deque<std::vector<int>> reduce_path(std::deque<std::vector<int>> input)
{

    for (int i = 0; i < input.size() - 1; ++i)
    {
        std::vector<int> start = input[i];

        if (i >= input.size() - 1)
        {
            break;
        }

        for (int j = i + 1; j < input.size() - 1; ++j)
        {
            std::vector<int> end = input[j];
            if (bresenham(start[0], start[1], end[0], end[1], false, true, tile_grid, false, 5))
            {
                input.erase(input.begin() + j);
                j--;
            }
            else
            {
                i = j - 1;
                break;
            }
        }
    }

    for (int i = 0; i < input.size(); ++i)
    {
        std::vector<int> start = input[i];
        float dx = float(start[0] * grid_div) - float(grid_offset);
        float dy = float(start[1] * grid_div) - float(grid_offset);
        cout << dx << " " << dy << endl;
    }

    cout << " end " << input.size() << endl;
    return input;
}

bool near_goal(int x, int y, int goal_x, int goal_y)
{
    if ((x - goal_x) * (x - goal_x) + (y - goal_y) * (y - goal_y) < 5)
    {
        return true;
    }
    return false;
}

// fix the guess
void fix_guess(Robot *robot)
{
    cout << "A" << endl;

    int x1 = std::floor((grid_offset + guess[0]) / grid_div);
    int y1 = std::floor((grid_offset + guess[1]) / grid_div);
    int range = 4;

    float w_range = 2.5;
    int window = std::ceil((w_range * 2 + grid_div) / grid_div);
    int window_shift = std::ceil(window / 2.0f);

    std::vector<std::vector<float>> relative_map(window, std::vector<float>(window, -1.0f));
    ;

    float r = robot->range;
    if (robot->range > 2.0)
    {
        r = 2.0001f;
    }

    float x2 = std::floor(r * cos(guess[2]) / grid_div);
    float y2 = std::floor(r * sin(guess[2]) / grid_div);

    bresenham_int(window_shift, window_shift, x2 + window_shift, y2 + window_shift, r <= 2.0f, false, relative_map, false, 1);

    /*
  for(const auto hit: robot->ranges) {
    // float r = hit.range;
    if (std::isinf(r) || r > 2) {
      r = 2.0001f;
    }

    float x2 = std::floor(r * cos(hit.angle + guess[2]) / grid_div);
    float y2 = std::floor(r * sin(hit.angle + guess[2]) / grid_div); 

    bresenham_int(window_shift, window_shift, x2 + window_shift, y2 + window_shift, r <= 2.0f, false, relative_map, false, 1);
  }
 */
    float shift_x = 0;
    float shift_y = 0;

    int best_x = 0;
    int best_y = 0;

    float min_err = FLT_MAX;
    float max_gain = -FLT_MAX;
    float min_err_none = 0.15;

    std::vector<float> c_col = coll;
    if (abs(time_delta) < 1000)
    {
        c_col = {0, 0, 0};
    }
    else
    {
        coll_delete = true;
    }

    for (int i1 = -range; i1 <= range; ++i1)
    {
        for (int i2 = -range; i2 <= range; ++i2)
        {
            float err = 0;
            float hit = 0;
            for (int j1 = 0; j1 < window; ++j1)
            {
                for (int j2 = 0; j2 < window; ++j2)
                {
                    int x_actual = x1 - window_shift + i1 + j1;
                    int y_actual = y1 - window_shift + i2 + j2;

                    if (x_actual < 0 || x_actual >= size || y_actual < 0 || y_actual >= size)
                    {
                        continue;
                    }

                    if (relative_map[j1][j2] == -1 || tile_grid[x_actual][y_actual] == -1)
                    {
                        continue;
                    }
                    else if ((relative_map[j1][j2] > 0) ^ (tile_grid[x_actual][y_actual] > 0))
                    {
                        err += abs(relative_map[j1][j2] * tile_grid[x_actual][y_actual]);
                    }
                    else
                    {
                        hit += abs(relative_map[j1][j2] * tile_grid[x_actual][y_actual]);
                    }
                }
            }

            float cum_err = abs((guess[0] + i1 * grid_div) - (last_guess[0] + c_col[0])) * predict_weight + abs((guess[1] + i2 * grid_div) - (last_guess[1] + c_col[1])) * predict_weight + err - (hit * hit_weight);

            if (hit > 0)
            {
                if (cum_err < min_err)
                {
                    best_x = i1;
                    best_y = i2;
                    min_err = cum_err;
                    max_gain = hit;
                }
            }
        }
    }

    if (abs(guess[0] - last_guess[0]) < 0.001)
    {
        best_x = 0;
    }
    if (abs(guess[1] - last_guess[1]) < 0.001)
    {
        best_y = 0;
    }

    offset[0] += best_x * grid_div;
    offset[1] += best_y * grid_div;

    offset[0] = clamp((-10.0f * 0.2f) / input_scalar, offset[0], (10.0f * 0.2f) / input_scalar);
    offset[1] = clamp((-10.0f * 0.2f) / input_scalar, offset[1], (10.0f * 0.2f) / input_scalar);

    last_guess = {robot->pos_x / input_scalar + offset[0], robot->pos_y / input_scalar + offset[1], robot->pos_t};

    if (abs(time_delta) > 1000)
    {
        guess_diff = {last_guess[0] - save_guess[0], last_guess[1] - save_guess[1], last_guess[2] - save_guess[2]};
        save_guess = {last_guess[0], last_guess[1], last_guess[2]};
    }
}

// command the robot to drive to the next point on the slam grid
bool drive_to_point(Robot *robot, std::vector<int> point)
{
    draw_point(float(point[0] * grid_div), float(point[1] * grid_div), 0.0f, 1.0f, 0.0f);
    float dx = (float(point[0] * grid_div) - grid_offset) - guess[0];
    float dy = (float(point[1] * grid_div) - grid_offset) - guess[1];
    float ang = atan2(dy, dx);
    float dist = dx * dx + dy * dy;
    float tang = ang - guess[2];

    if (tang > PI)
    {
        tang -= PI * 2;
    }
    else if (tang < -PI)
    {
        tang += PI * 2;
    }

    float c_dist = 0.35;
    // EDIT
    float fwd = clamp(0.0, robot->range, 2.0);

    //   float brgt = clamp(0.0, robot->ranges[0].range, 2.0);
    //   float rgt  = clamp(0.0, robot->ranges[1].range, 2.0);
    //   float frgt = clamp(0.0, robot->ranges[2].range, 2.0);
    //   float fwd  = clamp(0.0, robot->ranges[3].range, 2.0);
    //   float flft = clamp(0.0, robot->ranges[4].range, 2.0);
    //   float lft  = clamp(0.0, robot->ranges[5].range, 2.0);
    //   float blft = clamp(0.0, robot->ranges[6].range, 2.0);

    //   bool flag = brgt < c_dist || rgt  < c_dist || frgt < c_dist || fwd < 0.5 || flft < c_dist || lft < c_dist || blft < c_dist;

    bool flag = fwd < 0.5;
    //   std::vector<float> v = {brgt, rgt, frgt, fwd, flft, lft, blft};
    std::vector<float> v = {0, 0, cam_right, fwd, cam_left, 0, 0};

    float speed = 2.5f;
    float turn = 0.5f;

    if (dist < 2 * 2)
    {
        return true;
    }
    else if (tang < -0.20)
    {
        if (dist > 4 * 4 && tang > -0.35 && hit_drive)
        {
            cmd_vel(robot, speed + turn, speed - turn, false);
        }
        else
        {
            cmd_vel(robot, speed, -speed, false);
        }
        if (flag)
        {
            print_arr(v);
            cout << "cancel " << robot->range << endl;
            return true;
        }
        cout << "rotating " << ang << " " << guess[2] << " " << tang << endl;
        return false;
    }
    else if (tang > 0.20)
    {
        if (dist > 4 * 4 && tang < 0.35 && hit_drive)
        {
            cmd_vel(robot, speed - turn, speed + turn, false);
        }
        else
        {
            cmd_vel(robot, -speed, speed, false);
        }
        cout << "rotating " << ang << " " << guess[2] << " " << tang << endl;
        if (flag)
        {
            print_arr(v);
            cout << "cancel " << robot->range << endl;
            return true;
        }
        return false;
    }
    else if (dist > 2 * 2)
    {
        cout << "driving" << ang << " " << guess[2] << " " << tang << endl;
        hit_drive = true;
        if (flag)
        {
            print_arr(v);
            cout << "cancel " << robot->range << endl;
            return true;
        }
        cmd_vel(robot, speed, speed, false);
        return false;
    }

    hit_drive = false;

    return true;
}

void main_drive(Robot *robot)
{

    guess = {robot->pos_x / input_scalar + offset[0], robot->pos_y / input_scalar + offset[1], robot->pos_t};

    cout << coll[0] << " = " << coll[1] << " " << time_delta << endl;
    cout << save_guess[0] - guess[0] << " - " << save_guess[1] - guess[1] << " " << time_delta << endl;

    // fix_guess(robot);

    cout << "go" << endl;
    guess = {robot->pos_x / input_scalar + offset[0], robot->pos_y / input_scalar + offset[1], robot->pos_t};

    float r = robot->range;
    if (r > 2.1)
    {
        r = 2.0001f;
    }

    float x_val1 = guess[0] + clamp(0.0f, r, 2.0f) * cos(guess[2]);
    float y_val1 = guess[1] + clamp(0.0f, r, 2.0f) * sin(guess[2]);
    bresenham(guess[0], guess[1], x_val1, y_val1, r <= 2.0f, false, tile_grid, true, 1);

    // r = cam_left;
    // if (r > 2.1)
    // {
    //     r = 2.0001f;
    // }

    x_val1 = guess[0] + clamp(0.0f, r, 2.0f) * cos((PI/24) + guess[2]);
    y_val1 = guess[1] + clamp(0.0f, r, 2.0f) * sin((PI/24) + guess[2]);
    bresenham(guess[0], guess[1], x_val1, y_val1, r <= 2.0f, false, tile_grid, true, 1);

    // r = cam_right;
    // if (r > 2.1)
    // {
    //     r = 2.0001f;
    // }

    x_val1 = guess[0] + clamp(0.0f, r, 2.0f) * cos(-1 *(PI/24) + guess[2]);
    y_val1 = guess[1] + clamp(0.0f, r, 2.0f) * sin(-1 *(PI/24) + guess[2]);
    bresenham(guess[0], guess[1], x_val1, y_val1, r <= 2.0f, false, tile_grid, true, 1);

    /*
    for (const auto hit : robot->ranges)
    {
        float r = hit.range;
        if (std::isinf(r))
        {
            r = 2.0001f;
        }

        float x_val1 = guess[0] + clamp(0.0f, r, 2.0f) * cos(hit.angle + guess[2]);
        float y_val1 = guess[1] + clamp(0.0f, r, 2.0f) * sin(hit.angle + guess[2]);
        bresenham(guess[0], guess[1], x_val1, y_val1, r <= 2.0f, false, tile_grid, true, 1);
    }
    */

    float c_dist = 0.35;

    float fwd = clamp(0.0, robot->range, 2.0);

    // float brgt = clamp(0.0, robot->ranges[0].range, 2.0);
    // float rgt = clamp(0.0, robot->ranges[1].range, 2.0);
    // float frgt = clamp(0.0, robot->ranges[2].range, 2.0);
    // float fwd = clamp(0.0, robot->ranges[3].range, 2.0);
    // float flft = clamp(0.0, robot->ranges[4].range, 2.0);
    // float lft = clamp(0.0, robot->ranges[5].range, 2.0);
    // float blft = clamp(0.0, robot->ranges[6].range, 2.0);

    // bool flag = brgt < c_dist || rgt < c_dist || frgt < c_dist || fwd < 0.5 || flft < c_dist || lft < c_dist || blft < c_dist;
    // std::vector<float> v = {brgt, rgt, frgt, fwd, flft, lft, blft};

    float frgt = cam_right;
    float flft = cam_left;

    bool flag = fwd < 0.5;
    std::vector<float> v = {0, 0, cam_right, fwd, cam_left, 0, 0};

    for (int i = 0; i < target_path.size(); ++i)
    {
        float dx = float(target_path[i][0] * grid_div) - float(grid_offset);
        float dy = float(target_path[i][1] * grid_div) - float(grid_offset);
        cout << i << " " << target_path[i][0] << " " << target_path[i][1] << " " << dx << " " << dy << endl;
    }

    if (has_path)
    {
        float dx = float(target_point[0] * grid_div) - float(grid_offset);
        float dy = float(target_point[1] * grid_div) - float(grid_offset);
        int sx = std::floor((grid_offset + guess[0]) / grid_div);
        int sy = std::floor((grid_offset + guess[1]) / grid_div);

        cout << "driving to = [" << dx << ", " << dy << "] p_left: [" << target_path.size() << "] current = [" << guess[0] << " " << guess[1] << "] " << thread_pos[0] << " " << thread_pos[1] << endl;
        flag = flag || !bresenham_int(sx, sy, target_point[0], target_point[1], false, true, tile_grid, false, 0);
        if (tile_grid[target_point[0]][target_point[1]] > 0.5 || flag)
        {
            cout << "path_updated:" << tile_grid[target_point[0]][target_point[1]] << endl;
            print_arr(v);
            check_count2 = 0;
            has_path = false;
        }
        else if (drive_to_point(robot, target_point))
        {
            if (target_path.size() > 0)
            {
                target_point = target_path.back();
                target_path.pop_back();
            }
            else
            {
                target_point.clear();
                target_path.clear();
                check_count2 = 0;
                has_path = false;
            }
        }

        cout << "?" << endl;
        return;
    }
    else
    {
        cout << "path_cleared" << endl;
        has_path = false;

        target_point.clear();
        check_count2 = 0;
    }

    check_count2 = check_limit2;

    cout << "cc:" << check_count2 << " p?:" << has_path << " s?:" << searching << endl;
    check_count2++;

    float speed = 2.5f;
    float turn = 1.5f;
    float f_min = 0.6f;
    float f2_min = 0.75f;
    float f3_min = 0.9f;
    
    if (fwd < 1.2 || frgt < f_min && robot->range > 0.58 || flft < f_min && robot->range > 0.58)
    {
        // if (lft < 1.0 && flft > 1.8 || left_follow)
        if (flft > 1.8 || left_follow)
        {
            cout << "A1" << endl;
            cmd_vel(robot, -speed, speed, false);
        }
        // else if (rgt < 1.0 && frgt > 1.8 || right_follow)
        else if (frgt > 1.8 || right_follow)
        {
            cout << "A2" << endl;
            cmd_vel(robot, speed, -speed, false);
        }
        // else if (rgt + frgt > lft + flft)
        else if (int(left_follow) + frgt > int(right_follow) + flft)
        {
            cout << "B1" << endl;
            cmd_vel(robot, speed, -speed, false);
        }
        else
        {
            cout << "B2" << endl;
            cmd_vel(robot, -speed, speed, false);
        }
    }
    // else if (lft < 1.8 && lft > 1.1 && flft > lft)
    // {
    //     cout << "E" << endl;
    //     cmd_vel(robot, speed - turn, speed + turn, false);
    // }
    // else if ((flft / sqrt(2) < f2_min || lft < f2_min))
    else if ((flft / sqrt(2) < f2_min))
    {
        // if (flft / sqrt(2) < 0.6 && lft < flft / sqrt(2))
        if (flft / sqrt(2) < 0.6)
        {
            cout << "EEE" << endl;
            cmd_vel(robot, speed, speed, false);
        }
        // else if (lft / 1.3 > flft / sqrt(2) || flft / sqrt(2) < f_min)
        else if (flft / sqrt(2) < f_min)
        {
            cmd_vel(robot, speed + turn, speed - turn, false);
            cout << "F1" << endl;
        }
        // else if (lft * 1.3 < flft / sqrt(2) && flft < f3_min)
        else if (flft < f3_min)
        {
            cmd_vel(robot, speed - turn, speed + turn, false);
            cout << "F2" << endl;
        }
        else
        {
            cout << "EDAWED" << endl;
            cmd_vel(robot, speed, speed, false);
        }
    }
    // else if (rgt < 1.8 && rgt > 1.1 && frgt > rgt && blft > 1.7 && lft > 1.7 && flft > 1.7)
    // {
    //     cout << "C" << endl;
    //     cmd_vel(robot, speed + turn, speed - turn, false);
    // }
    // else if ((frgt / sqrt(2) < f2_min || rgt < f2_min) && blft > 1.7 && lft > 1.7 && flft > 1.7)
    else if ((frgt / sqrt(2) < f2_min || flft > 1.7))
    {
        // if (frgt / sqrt(2) < 0.6 && rgt < frgt / sqrt(2))
        if (frgt / sqrt(2) < 0.6)
        {
            cmd_vel(robot, speed, speed, false);
        }
        // else if (rgt / 1.3 > frgt / sqrt(2) || frgt / sqrt(2) < f_min)
        else if (frgt / sqrt(2) < f_min)
        {
            cmd_vel(robot, speed - turn, speed + turn, false);
            cout << "D1" << endl;
        }
        // else if (rgt * 1.3 < frgt / sqrt(2) && frgt < f3_min)
        else if (frgt < f3_min)
        {
            cmd_vel(robot, speed + turn, speed - turn, false);
            cout << "D2" << endl;
        }
        else
        {
            cmd_vel(robot, speed, speed, false);
        }
    }
    // else if (((blft < 1.5 && (lft > 1.5 || flft > 1.5)) || (lft < 1.3 && flft > 1.5)) && rgt > 1.2)
    // {
    //     cmd_vel(robot, speed - turn * 2, speed + turn, false);
    //     cout << "G" << endl;
    // }
    // else if (((brgt < 1.5 && (rgt > 1.5 || frgt > 1.5)) || (rgt < 1.3 && frgt > 1.5)) && lft > 1.2)
    // {
    //     cmd_vel(robot, speed + turn, speed - turn * 2, false);
    //     cout << "G2" << endl;
    // }
    else
    {
        cmd_vel(robot, speed, speed, false);
        cout << "fwd" << endl;
        // if (rgt > 1.8 && frgt > 1.8)
        if (frgt > 1.8)
        {
            right_follow = false;
        }
        // if (lft > 1.8 && flft > 1.8)
        if (flft > 1.8)
        {
            left_follow = false;
        }
    }

    
}

// main callback
void callback(Robot *robot)
{
    cout << "x " << robot->pos_x << endl << " y " << robot->pos_y << endl;
    cout << "here2" << endl;
    // cam_show(robot->frame);
    cout << "hit " << robot->range << endl;

    cout << "frame w " << robot->frame.cols << " h " << robot->frame.rows << endl;
    cout << "frame2 w " << robot->frame.size().width << " h " << robot->frame.size().height << endl;

    std::vector<std::vector<int>> fr_vals = get_im_vals(robot->frame);
    cout << "done" << endl;

    // cout << "frame v " << fr_vals[0][0] << " " << fr_vals[1][0] << " " << fr_vals[2][0] << endl;
    cout << "frame v ";
    double ccc = 300.0;

    cam_left = get_30_deg(mod_val(double(fr_vals[0][0])));
    cam_mid = mod_val(double(fr_vals[1][0]));
    cam_left = get_30_deg(mod_val(double(fr_vals[2][0])));
    cout << get_30_deg(mod_val(double(fr_vals[0][0]))) << " ";
    cout << mod_val(double(fr_vals[1][0])) << " ";
    cout << get_30_deg(mod_val(double(fr_vals[2][0]))) << " ";
    cout << endl;

    // if (robot->frame.cols > 0) {
    //     cv::Vec3b pixelColor = robot->frame.at<cv::Vec3b>(400,400);
    //     cout << "mid framasmefmw asd " << (int)pixelColor.val[0] << " " << (int)pixelColor.val[1] << " " << (int)pixelColor.val[2] << endl;
    // }

    robot->set_vel(0.0, 0.0);

    // EDIT
    time_delta = time(NULL) - last_time;
    last_time = time(NULL);

    if (setup_c < 20)
    {
        setup_c++;
        return;
    }

    at_goal = robot->at_goal();
    if (mx2.try_lock())
    {

        if (new_goal)
        {
            target_path.clear();

            for (int i = 0; i < target_path_thread.size(); ++i)
            {
                std::vector<int> p = {target_path_thread[i][0], target_path_thread[i][1]};
                target_path.emplace_back(p);
            }

            if (target_path.size() > 0)
            {
                target_point = target_path.back();
                target_path.pop_back();

                has_path = true;
            }

            new_goal = false;
        }

        for (int i = 0; i < size; ++i)
        {
            for (int i2 = 0; i2 < size; ++i2)
            {
                tile_grid_thread[i][i2] = tile_grid[i][i2];
            }
        }

        thread_pos = {guess[0], guess[1]};

        mx2.unlock();
    }

    main_drive(robot);
}

// add a point to a searching path
//  paths is the list of all paths
//  path_idx is the current index in paths that we are operating on
//  used if the path index leads to a new territorty set to true (makes it so that the parent method doesn't move on to the next path index)
//  not checked a tile grid of tiles that denotes ones that have and havent been checked yet
//  x,y the position to add
//  x2,y2 the goal
//  not_finished handles the edgecase where the current list is the last direction to be checked in this case we dont need to add a branching node at this point because its the only accesible direction
bool add_path(std::vector<std::deque<std::vector<int>>> *paths, int path_idx,
              bool *used, std::deque<std::vector<int>> *found_path,
              std::vector<std::vector<bool>> *not_checked, std::vector<std::vector<float>> &grid,
              int x, int y, int x2, int y2, bool not_finished)
{

    if (x < 0 || y < 0 || x >= size || y >= size)
    {
        return false;
    }

    if ((*not_checked)[x][y])
    {
        (*not_checked)[x][y] = false;
        draw_point((x) / grid_div, float(y) / grid_div, 0, 0, 1.0f);

        if (is_adj(x, y, 0, grid, 3) && !near_goal(x, y, x2, y2))
        {
            return false;
        }

        std::vector<int> point = {x, y};

        if (*used)
        {
            std::deque<std::vector<int>> new_path((*paths)[path_idx]);
            new_path.emplace_front(point);
            paths->emplace_back(new_path);

            if (near_goal(x, y, x2, y2) && grid[x][y] < 0)
            {
                *found_path = new_path;
                return true;
            }
        }
        else
        {

            if (not_finished)
            {
                std::deque<std::vector<int>> new_path((*paths)[path_idx]);
                paths->emplace_back(new_path);
            }

            (*paths)[path_idx].emplace_front(point);
            *used = true;

            if (near_goal(x, y, x2, y2) && grid[x][y] < 0)
            {
                *found_path = (*paths)[path_idx];
                return true;
            }
        }
    }

    return false;
}

// get a path to a given target using a flood-fill algorithm
std::deque<std::vector<int>> get_target(float xf, float yf, float t_x, float t_y, std::vector<std::vector<float>> &grid)
{
    std::vector<std::vector<bool>> not_checked(size, std::vector<bool>(size, true));
    std::vector<std::deque<std::vector<int>>> paths;

    int x1 = std::floor((grid_offset + xf) / grid_div);
    int y1 = std::floor((grid_offset + yf) / grid_div);
    int x2 = std::floor((grid_offset + t_x) / grid_div);
    int y2 = std::floor((grid_offset + t_y) / grid_div);
    std::deque<std::vector<int>> path_try = {{x1, y1}};

    paths.emplace_back(path_try);
    int path_idx = 0;
    std::deque<std::vector<int>> found_path = {{x1, y1}};

    while (path_idx < paths.size())
    {
        bool used = false;

        int xo = paths[path_idx][0][0];
        int yo = paths[path_idx][0][1];

        std::vector<std::vector<int>> n2 = {{xo, yo + 1}, {xo + 1, yo}, {xo, yo - 1}, {xo - 1, yo}};
        std::vector<int> order = {0, 1, 2, 3};
        std::vector<float> orderf = {0, 1, 2, 3};
        std::vector<float> dists = {0, 0, 0, 0};

        for (int i = 0; i < 3; ++i)
        {
            for (int i = 0; i < n2.size() - 1; ++i)
            {
                std::vector<int> p1 = n2[order[i]];
                std::vector<int> p2 = n2[order[i + 1]];

                float v1 = ((p1[0] - x2) * (p1[0] - x2)) + ((p1[1] - y2) * (p1[1] - y2));
                float v2 = ((p2[0] - x2) * (p2[0] - x2)) + ((p2[1] - y2) * (p2[1] - y2));

                dists[order[i]] = v1;
                dists[order[i + 1]] = v2;

                if (v1 > v2)
                {
                    int tmp = order[i];
                    order[i] = order[i + 1];
                    order[i + 1] = tmp;
                }
            }
        }

        for (int i = 0; i < 4; ++i)
        {
            orderf[i] = float(order[i]);
        }

        bool end = false;

        for (int i = 0; i < 4; ++i)
        {
            int idx = order[i];
            if (add_path(&paths, path_idx, &used, &found_path, &not_checked, grid, n2[idx][0], n2[idx][1], x2, y2, i < 3))
            {
                end = true;
                break;
            }
        }

        if (end)
        {
            break;
        }

        if (!used)
        {
            path_idx += 1;
        }
    }

    draw_path(found_path);
    return found_path;
}

// threaded path calulation
void calc_path()
{
    cout << "Thread spawned" << endl;
    while (!at_goal)
    {
        if (!new_goal && !has_path)
        {
            mx2.lock();

            searching = true;
            target_path_thread = reduce_path(get_target(thread_pos[0], thread_pos[1], 20, 0, tile_grid_thread));
            searching = false;

            new_goal = true;
            mx2.unlock();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void robot_thread(Robot *robot)
{
    std::thread first(calc_path);
    robot->do_stuff();
}

int main(int argc, char *argv[])
{
    // cout << "initializing camera" << endl;
    // cam_init();
    cout << "making robot" << endl;
    Robot robot(argc, argv, callback);
    std::thread rthr(robot_thread, &robot);

    return viz_run(argc, argv);
}

/*
NEW++++
To view the camera image in time, you could press CTRL-T in Gazebo
, choosing the Topic-"~/tankbot0/tankbot/camera_sensor/link/camera/image", 
then a Image Window will pop up, in order to view the Image in time.
*/

/*

void
callback(Robot* robot)
{
    cout << "here2" <<endl;
    cam_show(robot->frame);
    cout << "hit " << robot->range << endl;
    robot->set_vel(0.0, 0.0);
}

int
main(int argc, char* argv[])
{
    cout << "here1" <<endl;
    cam_init();

    cout << "making robot" << endl;
    Robot robot(argc, argv, callback);
    robot.do_stuff();

    return 0;
}
*/