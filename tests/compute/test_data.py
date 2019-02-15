import dgl.data as data

def test_sbm():
    ds = data.SBM(2, 10, 2, 0, 1)
    gs, lgs, g_deg, lg_deg, pm_pd= list(zip(*ds))
    print(gs, lgs, g_deg, lg_deg, pm_pd)

if __name__ == '__main__':
    test_sbm()
