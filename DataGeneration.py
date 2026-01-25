# preprocess.py
import os, re, json, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import cm
import argparse

warnings.simplefilter("ignore", category=FutureWarning)
offset = 20

# ======= label maps (unchanged) =======
cookActivities = {
    "cairo.txt": {"Other": offset, "Work": offset + 1, "Take_medicine": offset + 2, "Sleep": offset + 3,
                  "Leave_Home": offset + 4, "Eat": offset + 5, "Bed_to_toilet": offset + 6},
    "kyoto7.txt": {"Other": offset, "Work": offset + 1, "Sleep": offset + 2, "Relax": offset + 3,
                   "Personal_hygiene": offset + 4, "Cook": offset + 5, "Bed_to_toilet": offset + 6},
    "milan.txt": {"Other": offset, "Work": offset + 1, "Take_medicine": offset + 2, "Sleep": offset + 3, "Relax": offset + 4,
                  "Leave_Home": offset + 5, "Eat": offset + 6, "Cook": offset + 7, "Bed_to_toilet": offset + 8, "Bathing": offset + 9}}

mappingActivities = {
    "cairo.txt": {"_": "Other","R1wake": "Other","R2wake": "Other","Nightwandering": "Other","R1workinoffice": "Work",
                  "Laundry": "Work","R2takemedicine": "Take_medicine","R1sleep": "Sleep","R2sleep": "Sleep","Leavehome": "Leave_Home",
                  "Breakfast": "Eat","Dinner": "Eat","Lunch": "Eat","Bedtotoilet": "Bed_to_toilet"},
    "kyoto7.txt": {"R1_Bed_to_Toilet": "Bed_to_toilet","R2_Bed_to_Toilet": "Bed_to_toilet","Meal_Preparation": "Cook",
                   "R1_Personal_Hygiene": "Personal_hygiene","R2_Personal_Hygiene": "Personal_hygiene","Watch_TV": "Relax",
                   "R1_Sleep": "Sleep","R2_Sleep": "Sleep","Clean": "Work","R1_Work": "Work","R2_Work": "Work","Study": "Other",
                   "Wash_Bathtub": "Other","_": "Other"},
    "milan.txt": {"_": "Other","Master_Bedroom_Activity": "Other","Meditate": "Other","Chores": "Work","Desk_Activity": "Work",
                  "Morning_Meds": "Take_medicine","Eve_Meds": "Take_medicine","Sleep": "Sleep","Read": "Relax","Watch_TV": "Relax",
                  "Leave_Home": "Leave_Home","Dining_Rm_Activity": "Eat","Kitchen_Activity": "Cook","Bed_to_Toilet": "Bed_to_toilet",
                  "Master_Bathroom": "Bathing","Guest_Bathroom": "Bathing"}}

class HARDataset:
    def __init__(self, seqfile, imagefile, locfile, catfile, bin=1, time_aware=False):
        self.cookActivities = cookActivities
        self.mappingActivities = mappingActivities
        self.MAX = 0
        self.dataset = None
        self.dataname = None
        self.seqfile = seqfile
        self.imagefile = imagefile
        self.locfile = locfile
        self.catfile = catfile
        self.time_aware = time_aware
        self.sensor_set = None
        self.bin = bin

    # ---------- IO helpers ----------
    @staticmethod
    def _ensure(path): os.makedirs(path, exist_ok=True)

    # ---------- reading & splitting ----------
    def __read__(self, dataname="milan"):
        self.dataname = dataname
        self.MAX = len(self.cookActivities[self.dataname + '.txt'])
        ts, sensors, signal, acts = [], [], [], []
        activity = '_'  # running label
        with open(f'dataset/{dataname}.txt', 'rb') as f:
            for line in f.readlines():
                f_info = line.decode().split()
                if not ('.' in str(f_info[0]) + str(f_info[1])):
                    f_info[1] = f_info[1] + '.000000'
                ts.append(int(str(f_info[1])[:2]) / (24/self.bin))  # hour/24
                sensors.append(str(f_info[2]))
                signal.append(str(f_info[3]))
                if len(f_info) == 4:
                    acts.append(activity)
                else:
                    if len(f_info) != 6:
                        raise ValueError(f"Invalid line: {f_info}")
                    des = ''.join(f_info[4:])
                    if 'begin' in des:
                        activity = re.sub('begin', '', des).rstrip()
                        acts.append(activity)
                    if 'end' in des:
                        acts.append(activity)
                        activity = '_'
        acts = [self.mappingActivities[self.dataname + '.txt'][a] for a in acts]
        df = pd.DataFrame({'activities': acts, 'timestamp': ts, 'sensors': sensors, 'signal': signal})
        if dataname == "orange":
            df = df[df['sensors'].str.contains('global') == False]
        self.dataset = df[df.activities != "_"].reset_index(drop=True)

    def __getseq__(self):
        ds = self.dataset.copy()

        # fixed column orders per dataset
        if self.dataname == "cairo":
            sensor_set = ['M010','M015','M001','M012','M027','M023','M022','M020','M006','M007','M009','M024','M003','M005','M017',
                          'T001','M021','M018','T005','M026','T002','T004','M004','M002','T003','M016','M019','M013','M025','M011','M008','M014']
        elif self.dataname == "milan":
            sensor_set = ['M006','M024','M025','M011','M017','M005','M020','M010','M015','M013','M019','D001','M009','M007','M023',
                          'T002','M012','D002','M002','T001','M001','M016','M008','M022','M028','D003','M003','M004','M018','M014','M021','M027','M026']
        elif self.dataname == "kyoto7":
            sensor_set = ['M07','M20','D07','D12','M38','AD1-A','M25','M01','M19','M45','M12','L09','M16','D09','D14','L10','M48','M17','M22','M35',
                          'L11','M37','M46','M49','M43','D10','L06','M04','AD1-B','M11','M32','I03','M24','M27','M03','D03','M23','M26','M31','D15',
                          'M50','M34','M15','D08','M33','L12','M13','M09','M21','M51','M44','L04','M18','D05','M47','M05','M02','M41','M29','M39','M42',
                          'M36','M08','AD1-C','M10','M40','M06','M28','L13','M30','M14']
        elif self.dataname == "orange":
            sensor_set = self.dataset['sensors'].unique().tolist()
        else:
            raise ValueError("Unknown dataname")

        self.sensor_set = sensor_set[:]  # <-- keep order for freqcut_room
        # one-hot encode sensors
        eye = np.eye(len(sensor_set), dtype=int)
        enc = {s: eye[i] for i, s in enumerate(sensor_set)}
        ds["sensors"] = ds["sensors"].map(enc)

        # normalize signal to 0/1-ish
        sig_raw = set(ds["signal"])
        nums = []
        for v in sig_raw:
            try: nums.append(float(v))
            except: pass
        sig_map = {"OFF": 0, "ON": 1, "ABSENT": 0, "PRESENT": 1, "CLOSE": 0, "OPEN": 1}
        if nums:
            mn, mx = min(nums), max(nums)
            for v in sig_raw:
                try:
                    x = float(v)
                    sig_map[v] = 0 if mx == mn else (x - mn) / (mx - mn)
                except: pass
        ds["signal"] = ds["signal"].map(sig_map)

        # build windows by activity transitions
        Y = ds['activities'].tolist()
        X = []
       
        for i in range(len(Y)):
            row = ds.iloc[i]
            v = row["sensors"].tolist()
            v.append(row["timestamp"])
            v.append(row["signal"])
            X.append(v)

        break_points, YY = [], []
        for i in range(len(Y)):
            if i == 0:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[0]])
                break_points.append(0)
            elif Y[i] != Y[i - 1]:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[i]])
                break_points.append(i)

        XX = []
        for i in range(len(break_points)):
            if i == len(break_points) - 1: XX.append(X[break_points[i]:])
            else:                           XX.append(X[break_points[i]:break_points[i + 1]])

        # dump
        for idx in range(len(XX)):
            M = pd.DataFrame(XX[idx]); N = YY[idx]
            if not self.time_aware:
                HARDataset._ensure(f'{self.seqfile}/{self.dataname}/{N}')
                M.to_csv(f'{self.seqfile}/{self.dataname}/{N}/{N}_{idx}.csv', index=False, header=False)
            else:
                if idx <= len(XX) * 0.7:
                    HARDataset._ensure(f'{self.seqfile}_train/{self.dataname}/{N}')
                    M.to_csv(f'{self.seqfile}_train/{self.dataname}/{N}/{N}_{idx}.csv', index=False, header=False)
                else:
                    HARDataset._ensure(f'{self.seqfile}_test/{self.dataname}/{N}')
                    M.to_csv(f'{self.seqfile}_test/{self.dataname}/{N}/{N}_{idx}.csv', index=False, header=False)

    def __freqcut__(self, threshold=0.05):
        """sensor-based pruning."""
        def prune_dir(path):
            if not os.path.isdir(path): return
            for file in os.listdir(path):
                fp = os.path.join(path, file)
                df = pd.read_csv(fp, header=None)
                H = df.iloc[:, :-2].to_numpy()
                drop_list = []
                uni, count = np.unique(H, axis=0, return_counts=True)
                for i in range(len(count)):
                    if count[i] / max(count) <= threshold:
                        drop_list.append(uni[i])
                for vec in drop_list:
                    try:
                        k = np.where(vec == 1)
                        df.drop(df[df.iloc[:, int(k[0])] == 1].index, inplace=True)
                    except Exception:
                        pass
                df.to_csv(fp, header=False, index=False)

        for i in range(self.MAX):
            if not self.time_aware:
                prune_dir(f"{self.seqfile}/{self.dataname}/{i + 20}")
            else:
                prune_dir(f"{self.seqfile}_train/{self.dataname}/{i + 20}")
                prune_dir(f"{self.seqfile}_test/{self.dataname}/{i + 20}")

    def padding(self, max_length=1500):
        """
        Make every CSV (segment) have shape [max_length, D],
        where D = len(sensor_set) + 2 (timestamp, signal).
        Must be called AFTER __getseq__ (which sets self.sensor_set).
        """
        if not hasattr(self, "sensor_set") or self.sensor_set is None:
            raise RuntimeError("padding() requires self.sensor_set; run __getseq__() first.")
        D = len(self.sensor_set) + 2  # fixed feature width for this dataset/run

        def coerce_width(mat: np.ndarray) -> np.ndarray:
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            # fix width to D
            if mat.shape[1] < D:
                pad = np.zeros((mat.shape[0], D - mat.shape[1]), dtype=float)
                mat = np.concatenate([mat, pad], axis=1)
            elif mat.shape[1] > D:
                mat = mat[:, :D]
            return mat

        def pad_time(mat: np.ndarray) -> np.ndarray:
            """Pad/truncate along time dimension to max_length."""
            T = torch.from_numpy(mat)
            if T.shape[0] <= max_length:
                pad_top = max_length - T.shape[0]
                T = nn.ZeroPad2d((0, 0, pad_top, 0))(T)  # (left,right,top,bottom)
            else:
                T = T[-max_length:, :]
            return T.numpy()

        def pad_dir(root: str):
            if not os.path.isdir(root):
                return
            for fname in os.listdir(root):
                fp = os.path.join(root, fname)
                try:
                    # IMPORTANT: files were written without headers -> read with header=None
                    mat = pd.read_csv(fp, header=None).to_numpy(dtype=float)
                except pd.errors.EmptyDataError:
                    # create an empty stream with correct width; time padding will add zeros
                    mat = np.zeros((0, D), dtype=float)

                mat = coerce_width(mat)       # enforce width D
                mat = pad_time(mat)           # enforce time length
                pd.DataFrame(mat).to_csv(fp, index=False, header=False)

        # apply to all class folders
        for cls in range(20, 20 + self.MAX):
            if not self.time_aware:
                pad_dir(f'{self.seqfile}/{self.dataname}/{cls}')
            else:
                pad_dir(f'{self.seqfile}_train/{self.dataname}/{cls}')
                pad_dir(f'{self.seqfile}_test/{self.dataname}/{cls}')

    # ---------- visualizations ----------
    def temp(self):
        color_list = ['black','red',"blue",'yellow','gray','silver','rosybrown','lightcoral','brown','darkred',
                      'sienna','bisque','tan','orange','wheat','darkgoldenrod','gold','darkkhaki','olive','yellowgreen',
                      'lawngreen','darkseagreen','forestgreen','green','lime','springgreen','fuchsia','turquoise',
                      'lightseagreen','teal','deepskyblue','navy','violet','crimson']
        sensor_C_map = {}
        if self.dataname == "orange":
            with open("dataset/orange_sensor_to_C.json", "r") as f:
                sensor_C_map = json.load(f)

        for i in range(20, 20 + self.MAX):
            if not self.time_aware:
                HARDataset._ensure(f"{self.imagefile}/{self.dataname}/{i}")
                walk_root = f'{self.seqfile}/{self.dataname}/{i}'
                save_root = f'{self.imagefile}/{self.dataname}/{i}'
            else:
                for p in ["train", "test"]:
                    HARDataset._ensure(f"{self.imagefile}_{p}/{self.dataname}/{i}")
                walk_root = None  # handled inside loop below

            def render_dir(root, outdir):
                for file in os.listdir(root):
                    path = os.path.join(root, file)
                    df = np.array(pd.read_csv(path).iloc[:, :-2])
                    L = df.shape[0]
                    Y, C = [], []
                    for j in range(L):
                        try:
                            y = np.where(df[j, :] == 1)[0][0]; Y.append(y)
                        except: y = -1
                        if self.dataname == "cairo":
                            if y in [5, 11, 15, 18, 24]: C.append(1)
                            elif y == -1: pass
                            else: C.append(2)
                        elif self.dataname == "milan":
                            if y == -1: pass
                            elif y in [22, 32]: C.append(1)
                            elif y in [2, 24, 29]: C.append(3)
                            else: C.append(2)
                        elif self.dataname == "kyoto7":
                            if y == -1: pass
                            elif y in [17, 18, 67]: C.append(1)
                            elif y in [5, 19, 31, 34, 38, 50, 53, 59, 64]: C.append(3)
                            elif y in [0, 11, 13, 20, 23, 51, 66]: C.append(4)
                            elif y in [28]: C.append(5)
                            else: C.append(2)
                        elif self.dataname == "orange":
                            if y != -1:
                                sname = self.sensor_set[y].lower()
                                C_type = sensor_C_map.get(sname, 9)
                                C.append(C_type)
                                print(C_type)
                        

                    col = [color_list[i] for i in C]
                    X = list(range(len(Y)))
                    plt.scatter(X, Y, marker='o', c=col)
                    plt.ylim(0, 75 if self.dataname == "kyoto7" else 35)
                    plt.axis('off'); plt.xticks([]); plt.yticks([]); plt.box(False)
                    HARDataset._ensure(outdir)
                    plt.savefig(f'{outdir}/{os.path.splitext(file)[0]}.jpeg', bbox_inches='tight', pad_inches=0)
                    plt.close()

            if not self.time_aware:
                render_dir(walk_root, save_root)
            else:
                for p in ["train", "test"]:
                    render_dir(f'{self.seqfile}_{p}/{self.dataname}/{i}', f'{self.imagefile}_{p}/{self.dataname}/{i}')

    def spa(self):
        ds = self.dataset.copy()
        def ensure_all():
            if not self.time_aware:
                for i in range(self.MAX): HARDataset._ensure(f"{self.locfile}/{self.dataname}/{i + 20}")
                for i in range(self.MAX): HARDataset._ensure(f"{self.locfile}_diff/{self.dataname}/{i + 20}")
            else:
                for i in range(self.MAX):
                    HARDataset._ensure(f"{self.locfile}_train/{self.dataname}/{i + 20}")
                    HARDataset._ensure(f"{self.locfile}_test/{self.dataname}/{i + 20}")
        ensure_all()

        # prep segments
        X = ds.iloc[:, 2:4]; Y = ds.iloc[:, 0]
        break_point, YY, XX = [], [], []
        for i in range(len(Y)):
            if i == 0:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[0]]); break_point.append(0)
            elif Y[i] != Y[i - 1]:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[i]]); break_point.append(i)
        for i in range(len(break_point)):
            XX.append(X[break_point[i]:] if i == len(break_point) - 1 else X[break_point[i]:break_point[i + 1]])
        for i in range(len(XX)):
            XX[i] = XX[i][~XX[i].signal.isin(["OFF", "CLOSE", "ABSENT"])].reset_index(drop=True)["sensors"]

        # dots (unchanged hard-coded maps)
        if self.dataname == "milan":
            dots = {"M028":[162,359],"M021":[97,168],"M020":[162,359],"M025":[240,629],"M007":[466,174],"M026":[566,207],"M008":[676,340],
                    "M006":[758,111],"M005":[1043,39],"M004":[817,340],"M019":[530,461],"M009":[599,594],"M013":[420,560],"M018":[394,798],
                    "M017":[522,801],"M027":[970,753],"M024":[168,1046],"M011":[598,910],"M010":[714,911],"M016":[478,982],"M015":[455,1066],
                    "M014":[453,1170],"D003":[447,1225],"M012":[714,1099],"M023":[546,1107],"M022":[629,1143],"T001":[704,1135],"M002":[1155,996],
                    "M003":[923,1101],"M001":[1148,1176],"D001":[1159,1231],"D002":[1095,1095],"T002":[563,868]}
        elif self.dataname == "cairo":
            dots = {'M021':[315,240],'M007':[466,402],'M009':[259,361],'M016':[217,63],'M023':[487,202],'M022':[436,241],'M027':[606,117],
                    'T002':[88,421],'M012':[155,426],'M005':[451,353],'M017':[219,94],"M015":[108,145],'T001':[400,387],'M003':[383,399],
                    'M004':[577,343],'M020':[184,355],'M019':[164,125],'T005':[644,140],'M026':[619,89],'M014':[161,223],'M008':[330,350],
                    'M024':[606,247],'M013':[146,327],'M006':[348,352],'M010':[261,422],'M002':[384,452],'T003':[79,160],'M001':[504,455],
                    'M018':[230,160],'M025':[654,161],'T004':[325,214],'M011':[314,484]}
        elif self.dataname == "kyoto7":
            dots = {'M03':[940,269],'I03':[1824,896],'L06':[625,1026],'M16':[1468,650],'D09':[1327,697],'M40':[625,1113],'AD1-B':[1589,931],'M28':[316,702],
                    'M27':[120,704],'D10':[1325,790],'M20':[1263,1124],'M17':[1468,832],'L12':[560,765],'M49':[314,216],'M44':[335,477],'M18':[1469,985],
                    'M15':[1466,584],'M51':[1457,1156],'D07':[1602,841],'M13':[1520,268],'D14':[1619,930],'M09':[1321,435],'M11':[1318,115],'M39':[626,980],
                    'M42':[489,585],'M25':[922,1175],'M30':[480,792],'M07':[1126,440],'M12':[1522,114],'M45':[121,368],'L13':[561,762],'M34':[436,1154],
                    'M50':[318,370],'AD1-A':[1581,759],'M48':[314,92],'M46':[119,219],'M05':[1126,114],'M37':[589,701],'M43':[323,589],'M10':[1320,271],
                    'M41':[627,1171],'M01':[1085,591],'M24':[1086,1175],'M35':[436,1012],'L10':[1449,792],'M33':[256,1157],'M26':[926,1024],'D03':[460,761],
                    'M21':[1081,983],'M32':[259,1007],'D15':[1622,989],'M04':[946,119],'D05':[596,747],'M29':[484,700],'M08':[1267,505],'L11':[1258,1154],
                    'M36':[435,881],'L09':[994,1137],'M23':[1084,647],'M22':[1083,831],'AD1-C':[1590,983],'M14':[1516,434],'M38':[625,825],'M19':[1270,984],
                    'M02':[943,439],'M06':[1126,275],'D12':[1177,695],'L04':[235,1207],'M31':[263,872],'M47':[123,92],'D08':[1331,652]}
        elif self.dataname == "orange":
            dots = pd.read_csv("./orange4home_2/sensor_coordinates.csv").drop(columns=["note"])
            dots.dropna(inplace=True)
            dots = dots.set_index("sensor")
            dots = dots.to_dict(orient="index")
            dots = {k: (v["x"], v["y"]) for k, v in dots.items()}
        else:
            raise ValueError("Unknown dataname")

        for i in range(len(YY)):
            label = YY[i]; series = XX[i]
            x, y, c = [], [], []
            fre = series.value_counts()
            for s in series:
                d = dots.get(s)
                if d is None: 
                    continue
                if self.dataname == "milan":
                    xx, yy = int(d[0] * 224 / 1100), int(d[1] * 224 / 1100)
                elif self.dataname == "cairo":
                    xx, yy = int(d[0] * 224 / 708), int(d[1] * 224 / 536)
                else:
                    xx, yy = int(d[0] * 224 / 1862), int(d[1] * 224 / 1280)
                x.append(xx); y.append(yy); c.append(cm.OrRd(fre[s] / max(fre)))

            if len(x) == 1:
                plt.plot(x, y, c=c[0], marker='o')
            elif len(x) > 1:
                for j in range(len(x) - 2):
                    plt.plot(x[j:j + 2], y[j:j + 2], lw=1, c=c[j], marker='o')
                plt.plot(x[-2:], y[-2:], lw=1, c=c[-2], marker='o')
            plt.xlim(0, 224); plt.ylim(0, 224); plt.axis('off')
            ax = plt.gca()
            for s in ['top','right','left','bottom']: ax.spines[s].set_visible(False)
            if not self.time_aware:
                plt.savefig(f"{self.locfile}/{self.dataname}/{label}/{label}_{i}.jpeg")
            else:
                if i <= len(YY) * 0.7:
                    plt.savefig(f"{self.locfile}_train/{self.dataname}/{label}/{label}_{i}.jpeg")
                else:
                    plt.savefig(f"{self.locfile}_test/{self.dataname}/{label}/{label}_{i}.jpeg")
            plt.close()

    def cat(self):
        def stitch(pair1_dir, pair2_dir, out_dir):
            if not os.path.isdir(pair1_dir) or not os.path.isdir(pair2_dir): return
            imgs1, imgs2, order = [], [], []
            for file in os.listdir(pair1_dir):
                path = os.path.join(pair1_dir, file)
                order.append(file); imgs1.append(np.array(Image.open(path).resize((224, 224))))
            for file in os.listdir(pair2_dir):
                path = os.path.join(pair2_dir, file)
                imgs2.append(np.array(Image.open(path).resize((224, 224))))
            for k in range(len(imgs1)):
                im = Image.fromarray(np.concatenate((imgs1[k], imgs2[k]), axis=0))
                HARDataset._ensure(out_dir)
                im.save(os.path.join(out_dir, f"{os.path.splitext(order[k])[0]}.jpeg"))

        for i in range(20, 20 + self.MAX):
            if not self.time_aware:
                stitch(f'{self.imagefile}/{self.dataname}/{i}',
                       f'{self.locfile}/{self.dataname}/{i}',
                       f'{self.catfile}/{self.dataname}/{i}')
            else:
                for p in ["train", "test"]:
                    stitch(f'{self.imagefile}_{p}/{self.dataname}/{i}',
                           f'{self.locfile}_{p}/{self.dataname}/{i}',
                           f'{self.catfile}_{p}/{self.dataname}/{i}')
        
    def plot_activity_lengths(self, save_png=None, per_label=False):
        """
        Plot segment length (number of rows / time stamps) vs segment index.
        If per_label=True, makes one plot per activity label id.
        Call this right AFTER __getseq__(). If you want lengths AFTER cleaning,
        call it after __offdelete__() and recompute from files (see alt below).
        """
        # Reconstruct segments like in __getseq__
        ds = self.dataset.copy()
        Y = ds['activities'].tolist()
        break_points, YY = [], []
        for i in range(len(Y)):
            if i == 0:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[0]])
                break_points.append(0)
            elif Y[i] != Y[i - 1]:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[i]])
                break_points.append(i)
        seg_lens = []
        for i in range(len(break_points)):
            start = break_points[i]
            end   = break_points[i + 1] if i < len(break_points) - 1 else len(Y)
            seg_lens.append(end - start)

        if not per_label:
            plt.figure(figsize=(8, 3))
            plt.plot(range(len(seg_lens)), seg_lens, marker='o', linestyle='none', ms=2)
            plt.xlabel("Segment index")
            plt.ylabel("# time stamps")
            plt.title(f"{self.dataname}: segment lengths (N={len(seg_lens)})")
            plt.tight_layout()
            if save_png:
                HARDataset._ensure(os.path.dirname(save_png))
                plt.savefig(save_png, dpi=200)
                plt.close()
            else:
                plt.show()
        else:
            # one figure per label id
            from collections import defaultdict
            by_lab = defaultdict(list)
            for idx, (lab, L) in enumerate(zip(YY, seg_lens)):
                by_lab[lab].append((idx, L))
            for lab, arr in by_lab.items():
                x = [a[0] for a in arr]; y = [a[1] for a in arr]
                plt.figure(figsize=(8, 3))
                plt.plot(x, y, marker='o', linestyle='none', ms=2)
                plt.xlabel("Segment index")
                plt.ylabel("# time stamps")
                plt.title(f"{self.dataname}: label {lab} segment lengths (N={len(y)})")
                plt.tight_layout()
                if save_png:
                    out = save_png.replace(".png", f"_{lab}.png")
                    HARDataset._ensure(os.path.dirname(out))
                    plt.savefig(out, dpi=200); plt.close()
                else:
                    plt.show()
                    
    def plot_activity_lengths_bar(self, save_png=None, per_label=False, logy=False):
        """
        Bar plot: segment index (x) vs number of time stamps (y).
        Call right AFTER __getseq__() if you want pre-clean lengths.
        Set per_label=True to emit one bar plot per activity label id.
        """
        ds = self.dataset.copy()
        Y = ds['activities'].tolist()

        # find segment boundaries and labels
        break_points, YY = [], []
        for i in range(len(Y)):
            if i == 0:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[0]])
                break_points.append(0)
            elif Y[i] != Y[i - 1]:
                YY.append(self.cookActivities[self.dataname + '.txt'][Y[i]])
                break_points.append(i)

        # segment lengths
        seg_lens = []
        for i in range(len(break_points)):
            start = break_points[i]
            end   = break_points[i + 1] if i < len(break_points) - 1 else len(Y)
            seg_lens.append(end - start)

        if not per_label:
            x = range(len(seg_lens))
            plt.figure(figsize=(10, 3))
            plt.bar(x, seg_lens, width=1.0)
            if logy: plt.yscale("log")
            plt.xlabel("Segment index")
            plt.ylabel("# time stamps")
            plt.title(f"{self.dataname}: segment lengths (N={len(seg_lens)})")
            plt.tight_layout()
            if save_png:
                plt.savefig(save_png, dpi=200); plt.close()
            else:
                plt.show()
        else:
            from collections import defaultdict
            by_lab = defaultdict(list)
            for idx, (lab, L) in enumerate(zip(YY, seg_lens)):
                by_lab[lab].append((idx, L))
            for lab, arr in by_lab.items():
                x = [a[0] for a in arr]; y = [a[1] for a in arr]
                plt.figure(figsize=(10, 3))
                plt.bar(x, y, width=1.0)
                if logy: plt.yscale("log")
                plt.xlabel("Segment index")
                plt.ylabel("# time stamps")
                plt.title(f"{self.dataname}: label {lab} segment lengths (N={len(y)})")
                plt.tight_layout()
                if save_png:
                    out = save_png.replace(".png", f"_{lab}.png")
                    _ensure(os.path.dirname(out))
                    plt.savefig(out, dpi=200); plt.close()
                else:
                    plt.show()

    # ---------- Corrupt -------------
    def seq_corrupt(self, ratio=0.01, ac=5):
        # Stimulate the sensor mulfunction. Please used before calling the padding function.
        k=0
        for i in range(self.MAX):
            path=f"{self.seqfile}/{self.dataname}/{i+20}"
            os.makedirs(path,exist_ok=True)
            for file in os.listdir(path):
                a=random.randint(0,99)
                if a<ac: # activity sample selected
                    k+=1
                    dir=os.path.join(path,file)
                    data=pd.read_csv(dir,header=None)
                    num_rows_to_drop = int(len(data) * ratio)
                    num_rows_to_drop = max(num_rows_to_drop, 0)
                    indices_to_drop = np.random.choice(data.index, size=num_rows_to_drop, replace=False)
                    data = data.drop(indices_to_drop)
                    data.to_csv(dir,header=False,index=False)
    
    def img_corrupt(self, var=10):
        # Stimulate the sensor mulfunction. Please used before calling the cat function.
        dataset=self.dataset.copy()
        for i in range(self.MAX):
            os.makedirs(f"{self.locfile}/{self.dataname}/{i+20}",exist_ok=True)
        
        rows = dataset.shape[0]
        temp = []
        l = []
        for i in range(rows):
            if 'T' in dataset.iloc[i, 2] or 'A' in dataset.iloc[i, 2]:
                try:
                    temp.append(float(dataset.iloc[i, 3]))
                    l.append(dataset.iloc[i, 0])
                except ValueError:
                    temp.append(float(dataset.iloc[i, 3][:-1]))
        
        X=dataset.iloc[:,2:4]
        Y=dataset.iloc[:,0]
        break_point = []
        length = len(Y)
        XX = []
        YY = []

        for i in range(length):
            if i == 0:
                YY.append(cookActivities[self.dataname+'.txt'][Y[0]])
                break_point.append(0)
                continue
            elif Y[i] != Y[i - 1]:
                YY.append(cookActivities[self.dataname+'.txt'][Y[i]])
                break_point.append(i)

        length = len(break_point)
        for i in range(length):
            if i == length - 1:
                XX.append(X[break_point[i]:])
            else:
                XX.append(X[break_point[i]:break_point[i + 1]])
        
        # drop OFF and CLOSE lines
        
        for i in range(length):
            XX[i] = XX[i].drop(XX[i][XX[i].signal=="OFF"].index)
            XX[i] = XX[i].drop(XX[i][XX[i].signal=="CLOSE"].index)
            XX[i] = XX[i].drop(XX[i][XX[i].signal=="ABSENT"].index).reset_index(drop=True)
            XX[i]=XX[i].sensors
        
        #define dots
        if self.dataname=="milan":
            dots={"M028":[162,359],"M021":[97,168],"M020":[162,359],"M025":[240,629],"M007":[466,174],"M026":[566,207],"M008":[676,340],
                "M006":[758,111],"M005":[1043,39],"M004":[817,340],"M019":[530,461],"M009":[599,594],"M013":[420,560],"M018":[394,798],
                "M017":[522,801],"M027":[970,753],"M024":[168,1046],"M011":[598,910],"M010":[714,911],"M016":[478,982],"M015":[455,1066],
                "M014":[453,1170],"D003":[447,1225],"M012":[714,1099],"M023":[546,1107],"M022":[629,1143],"T001":[704,1135],"M002":[1155,996],
                "M003":[923,1101],"M001":[1148,1176],"D001":[1159,1231],"D002":[1095,1095],"T002":[563,868]}
        elif self.dataname=="cairo":
            dots={'M021':[315,240], 'M007':[466,402], 'M009':[259,361], 'M016':[217,63], 'M023':[487,202], 'M022':[436,241], 'M027':[606,117],
                    'T002':[88,421], 'M012':[155,426], 'M005':[451,353], 'M017':[219,94], "M015":[108,145], 'T001':[400,387], 'M003':[383,399],
                    'M004':[577,343], 'M020':[184,355], 'M019':[164,125], 'T005':[644,140], 'M026':[619,89], 'M014':[161,223], 'M008':[330,350],
                    'M024':[606,247], 'M013':[146,327], 'M006':[348,352], 'M010':[261,422], 'M002':[384,452], 'T003':[79,160], 'M001':[504,455],
                    'M018':[230,160], 'M025':[654,161], 'T004':[325,214], 'M011':[314,484]}
        elif self.dataname=="kyoto7":
            dots={'M03':[940,269], 'I03':[1824,896], 'L06':[625,1026], 'M16':[1468,650], 'D09':[1327,697], 'M40':[625,1113], 'AD1-B':[1589,931], 'M28':[316,702],
                'M27':[120,704], 'D10':[1325,790], 'M20':[1263,1124], 'M17':[1468,832], 'L12':[560,765], 'M49':[314,216], 'M44':[335,477], 'M18':[1469,985], 
                'M15':[1466,584], 'M51':[1457,1156], 'D07':[1602,841], 'M13':[1520,268], 'D14':[1619,930], 'M09':[1321,435], 'M11':[1318,115], 'M39':[626,980], 
                'M42':[489,585], 'M25':[922,1175], 'M30':[480,792], 'M07':[1126,440], 'M12':[1522,114], 'M45':[121,368], 'L13':[561,762], 'M34':[436,1154], 
                'M50':[318,370], 'AD1-A':[1581,759], 'M48':[314,92], 'M46':[119,219], 'M05':[1126,114], 'M37':[589,701], 'M43':[323,589], 'M10':[1320,271], 
                'M41':[627,1171], 'M01':[1085,591], 'M24':[1086,1175], 'M35':[436,1012], 'L10':[1449,792], 'M33':[256,1157], 'M26':[926,1024], 'D03':[460,761], 
                'M21':[1081,983], 'M32':[259,1007], 'D15':[1622,989], 'M04':[946,119], 'D05':[596,747], 'M29':[484,700], 'M08':[1267,505], 'L11':[1258,1154], 
                'M36':[435,881], 'L09':[994,1137], 'M23':[1084,647], 'M22':[1083,831], 'AD1-C':[1590,983], 'M14':[1516,434], 'M38':[625,825], 'M19':[1270,984], 
                'M02':[943,439], 'M06':[1126,275], 'D12':[1177,695], 'L04':[235,1207], 'M31':[263,872], 'M47':[123,92], 'D08':[1331,652]}
        for key, value in dots.items():
            dots[key][0]+= np.random.normal(0, np.sqrt(var))
            dots[key][1]+= np.random.normal(0, np.sqrt(var))

        for i in range(len(YY)):
            #define trace
            label=YY[i]
            x=[]
            y=[]
            fre=XX[i].value_counts()
            c=[] # color
            for j in XX[i]:
                if self.dataname=="milan":
                    x.append(int(dots[j][0]*224/1100))
                    y.append(int(dots[j][1]*224/1100))
                    c.append(cm.OrRd(fre[j]/max(fre)))
                elif self.dataname=="cairo":
                    x.append(int(dots[j][0]*224/708))
                    y.append(int(dots[j][1]*224/536))
                    c.append(cm.OrRd(fre[j]/max(fre)))
                elif self.dataname=="kyoto7":
                    x.append(int(dots[j][0]*224/1862))
                    y.append(int(dots[j][1]*224/1280))
                    c.append(cm.OrRd(fre[j]/max(fre)))
            if len(x)==1:
                plt.plot(x, y, c=c[0], marker='o')
            elif len(x)==0:
                plt.plot()
            else:
                for j in range(len(x)-2):
                    plt.plot(x[j:(j+2)], y[j:(j+2)],lw=1, c=c[j], marker='o')
                plt.plot(x[-2:], y[-2:],lw=1, c=c[-2], marker='o')
            plt.xlim(0,256)
            plt.ylim(0,256)
            plt.savefig(f"{self.locfile}/{self.dataname}/{label}/{label}_{i}.jpeg")
            plt.close()


# ---------- CLI pipeline ----------
def _tag(mode: str, thr: float) -> str:
    """Folder tag that encodes filter type + percentage."""
    t = int(round(thr * 100))
    return f"_sens_t{t:02d}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=["milan","cairo","kyoto7"])
    parser.add_argument("--mode", type=str, choices=["sensor"], default="sensor")
    parser.add_argument("--threshold", type=float, default=0.00, help="freqcut threshold")
    parser.add_argument("--time-aware", action="store_true", default=False, help="Setting True for autoregressive experiment")
    parser.add_argument("--maxlen", type=int, default=1500, help="Padding length")
    parser.add_argument("--bin", type=float, default=1, help="Temporal Binning")

    parser.add_argument("--seq_corrupt", action="store_true", help="Setting True to stimult the sensor malfunction")
    parser.add_argument("--img_corrupt", action="store_true", help="Setting True to stimult the sensor malfunction")
    parser.add_argument("--ratio", type=int, default=0.05, help="The proportion of data dropped within a single activity")
    parser.add_argument("--ac", type=int, default=5, help="The possibility of whether sensor malfunction occurred for a single activity")
    parser.add_argument("--var", type=int, default=10, help="The variance of physical sensor movement")
    args = parser.parse_args()

    tag = _tag(args.mode, args.threshold)

    # default (sensor) keeps I/O folders unchanged
    if args.seq_corrupt:
        seq_root   = f"datasets/seq{tag}_ratio{args.ratio}_ac{args.ac}"
        loc_root   = f"datasets/loc{tag}_ratio{args.ratio}_ac{args.ac}"
        img_root   = f"datasets/image{tag}_ratio{args.ratio}_ac{args.ac}"
        cat_root   = f"datasets/cat{tag}_ratio{args.ratio}_ac{args.ac}"
    elif args.img_corrupt:
        seq_root   = f"datasets/seq{tag}_var{args.var}"
        loc_root   = f"datasets/loc{tag}_var{args.var}"
        img_root   = f"datasets/image{tag}_var{args.var}"
        cat_root   = f"datasets/cat{tag}_var{args.var}"
    else:
        seq_root   = f"datasets/seq{tag}"
        loc_root   = f"datasets/loc{tag}"
        img_root   = f"datasets/image{tag}"
        cat_root   = f"datasets/cat{tag}"

    for name in args.datasets:
        H = HARDataset(seqfile=seq_root, locfile=loc_root, imagefile=img_root, catfile=cat_root, bin=args.bin, time_aware=args.time_aware)
        print("Reading data...")
        H.__read__(dataname=name)
        print("Getting sequence...")
        H.__getseq__()
        print("Frequency cutting...")
        H.__freqcut__(threshold=args.threshold)

        if args.seq_corrupt:
            H.seq_corrupt(args.ratio, args.ac)

        print("Padding...")
        H.padding(max_length=args.maxlen)
        print("Generating Temperal Images...")
        H.temp();

        if args.img_corrupt:
            H.img_corrupt(args.var)
        else:
            print("Generating Spatial Images...")
            H.spa();
        
        print("Generating Category Images...")
        H.cat();
        print("Done!")



