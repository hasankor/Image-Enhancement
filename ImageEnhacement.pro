QT += quick quickwidgets

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += /home/hasankor/Documents/opencv-4.x/install/lib/pkgconfig/opencv4.pc

#INCLUDEPATH +=/usr/local/include
#INCLUDEPATH +=/usr/local/cuda/include

#LIBS += -L/usr/local/cuda/lib64  -lcudart -lnppc -lcuda -lcurand -lcudadevrt -lcudart_static -lcublas
#LIBS += -ldl -lm -lrt

    INCLUDEPATH             +=  /usr/include/gstreamer-1.0 \
                                /usr/include/glib-2.0 \

    unix:linux:contains(QT_ARCH, arm64) {
            INCLUDEPATH     +=  /usr/lib/aarch64-linux-gnu/glib-2.0/include /
    }else{
            INCLUDEPATH     +=  /usr/lib/x86_64-linux-gnu/glib-2.0/include /
    }

    LIBS                    +=  -lgstreamer-1.0 -lglib-2.0 -lgobject-2.0 -lgstapp-1.0

SOURCES += \
        src/autobrightnessconstrast/autobrightnessandcontrast.cpp \
        src/defogger16bit/defogger16bit.cpp \
        src/badpixelreplacement/badpixelreplacement.cpp \
        src/defogger/defogger.cpp \
        src/hazeremove/hazeremove.cpp \
        main.cpp \
        src/nuc/nonuniformcorrection.cpp \
        src/stretch/cpustretcher.cpp

HEADERS += \
        backup/Test.h \
        backup/simplestcolorbalance.h \
        src/autobrightnessconstrast/autobrightnessandcontrast.h \
        src/defogger16bit/defogger16bit.h \
        src/badpixelreplacement/badpixelreplacement.h \
        src/defogger/defogger.h \
        src/hazeremove/hazeremove.h \
        src/nuc/nonuniformcorrection.h \
        src/Common.h \ \
    src/stretch/cpustretcher.h


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

