import config as cfg

from data.preprocessing.h36m_train import h36m_train_extract
from data.preprocessing.h36m_valid import h36m_extract
from data.preprocessing.mpi_inf_3dhp import train_data


def extracting_image():
    h36m_train_extract(cfg.H36M_ROOT)
    h36m_extract(cfg.H36M_ROOT, protocol=1)
    h36m_extract(cfg.H36M_ROOT, protocol=2)

    train_data(cfg.MPI_INF_3DHP_ROOT)


if __name__ == "__main__":
    extracting_image()