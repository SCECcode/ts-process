FC=gfortran
FFLAGS = -O2 -ffixed-line-length-none
HEADS = baseline.h
COMMON_OBJS = calcrsp.o fftsub.o ft_th.o sort.o spline.o splint.o
ROTD50_OBJS = ${COMMON_OBJS} rotd50.o

all: rotd50

rotd50: ${ROTD50_OBJS}
	${FC} ${FFLAGS} -o rotd50 ${ROTD50_OBJS}

${OBJS}: ${HEADS}

clean:
	rm -f ${ROTD50_OBJS} rotd50 *~
