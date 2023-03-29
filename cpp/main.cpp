#include <iostream>
#include "detector.cpp"
#include "classifier.cpp"

int main(int argc, const char* argv[]) {

    std::string model = "";
    if (argc > 1) {
        model = argv[1];
    }
    
    ObjectDetector* detector;
    if (model == "ssdlitemobilenet") {
        detector = new SSDLiteMobileNet();
    } 
    else if (model == "fasterrcnnmobilenet") {
        detector = new FasterRCNNMobileNet();
    } else {
        detector = new EdgeFormer();
    }
    detector->initModel();
    detector->run();
    detector->close();
    return 0;
}