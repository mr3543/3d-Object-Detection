#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PillarPoint
{
    public:
        PillarPoint(double x,double y,double z,
                    double r, double canvas_x, double canvas_y);

        void make_feature(py::array_t<double> &tensor,
                          int pillar_ind,
                          int point_ind) const;
        void set_xc(double xc);
        void set_yc(double yc);
        void set_zc(double zc);
        double get_x() const;
        double get_y() const;
        double get_z() const;
    private:
        double x;
        double y;
        double z;
        double r;
        double xp;
        double yp;
        double canvas_x;
        double canvas_y;
        double xc;
        double yc;
        double zc;
};

class Pillar
{
    public:
        Pillar(double canvas_x,double canvas_y);
        void add_point(PillarPoint *pp);
        std::vector<double> point_mean() const;
        std::vector<PillarPoint*> &get_points();

    private:
        std::vector<PillarPoint*> points;
        double canvas_x;
        double canvas_y;
};
