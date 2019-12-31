#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PillarPoint
{
    /*
        Pillar point class. Represents a single lidar point. Class 
        contains the x,y,z,r coordinates of the point and the x,y
        coordinates of the pillar it belongs to. 

        The method make_feature extracts the feature for the point 
        which is grouped together with the other points to make the 
        net work input tensor.
    */
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
    /*
        Pillar class. This contains a vector of pillar points
        which belong the the pillar. Methods are provided to add
        a point to the pillar and to get the mean (x,y,z) coordinates
        of the points in the pillar. 

    */
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
