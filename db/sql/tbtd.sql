BEGIN TRANSACTION;
CREATE TABLE epics_channel(
  id integer primary key,
  lcs_name text unique,
  value_type text,  -- current or voltage
  value double precision default 0.0,
  thresh double precision default 0.0,
  value_txt text default 'NA',
  update_time date --default datetime('now')
);
INSERT INTO "epics_channel" VALUES(1, 'TBQL001V01', 'DVM', 8.242, 0.05, 'NA', '2013-04-24 11:51:16');
INSERT INTO "epics_channel" VALUES(2, 'TBQL001V02', 'DVM', -25.065, 0.05, 'NA', '2013-04-24 11:51:23');
INSERT INTO "epics_channel" VALUES(3, 'TBQL001V03', 'DVM', 20.605, 0.05, 'NA', '2013-04-24 11:51:29');
INSERT INTO "epics_channel" VALUES(4, 'TBQL002V01', 'DVM', -24.996, 0.05, 'NA', '2013-04-24 11:51:36');
INSERT INTO "epics_channel" VALUES(5, 'TBQL002V02', 'DVM', 36.866, 0.05, 'NA', '2013-04-24 11:51:39');
INSERT INTO "epics_channel" VALUES(6, 'TBQL002V03', 'DVM', -19.656, 0.05, 'NA', '2013-04-24 11:51:43');
INSERT INTO "epics_channel" VALUES(7, 'TBQL003V01', 'DVM', -3.006, 0.05, 'NA', '2013-04-24 11:51:49');
INSERT INTO "epics_channel" VALUES(8, 'TBQL003V02', 'DVM', 1.644, 0.05, 'NA', '2013-04-24 11:51:52');
INSERT INTO "epics_channel" VALUES(9, 'TBQL004V01', 'DVM', -0.642, 0.05, 'NA', '2013-04-24 11:51:58');
INSERT INTO "epics_channel" VALUES(10, 'TBQL004V02', 'DVM', 0.0, 0.05, 'NA', '2013-04-24 11:52:10');
INSERT INTO "epics_channel" VALUES(11, 'TBQL005V01', 'DVM', 2.335, 0.05, 'NA', '2013-04-24 11:52:22');
INSERT INTO "epics_channel" VALUES(12, 'TBQL005V02', 'DVM', -33.178, 0.05, 'NA', '2013-04-24 11:52:27');
INSERT INTO "epics_channel" VALUES(13, 'TBQL005V03', 'DVM', 55.494, 0.05, 'NA', '2013-04-24 11:52:32');
INSERT INTO "epics_channel" VALUES(14, 'TBQL005V04', 'DVM', -27.783, 0.05, 'NA', '2013-04-24 11:52:38');
INSERT INTO "epics_channel" VALUES(15, 'TBQL006V01', 'DVM', 9.89, 0.05, 'NA', '2013-04-24 11:52:46');
INSERT INTO "epics_channel" VALUES(16, 'TBQL006V02', 'DVM', -33.914, 0.05, 'NA', '2013-04-24 11:52:52');
INSERT INTO "epics_channel" VALUES(17, 'TBQL006V03', 'DVM', 43.169, 0.05, 'NA', '2013-04-24 11:52:57');
INSERT INTO "epics_channel" VALUES(18, 'TBQL006V04', 'DVM', -20.715, 0.05, 'NA', '2013-04-24 11:53:03');
INSERT INTO "epics_channel" VALUES(19, 'TDQL001V01', 'DVM', -30.806, 0.05, 'NA', '2013-04-24 11:53:14');
INSERT INTO "epics_channel" VALUES(20, 'TDQL001V02', 'DVM', 57.397, 0.05, 'NA', '2013-04-24 11:53:19');
INSERT INTO "epics_channel" VALUES(21, 'TDQL001V03', 'DVM', -73.37, 0.05, 'NA', '2013-04-24 11:53:23');
INSERT INTO "epics_channel" VALUES(22, 'TDQL001V04', 'DVM', 64.952, 0.05, 'NA', '2013-04-24 11:53:30');
INSERT INTO "epics_channel" VALUES(23, 'TBDB002E04', 'buncher_amp', 40.7, 0.0, 'NA', '2016-03-29 10:42:00');
INSERT INTO "epics_channel" VALUES(24, 'TBDB002E02', 'buncher_ph', 270.0, 0.0, 'NA', '2016-03-29 10:42:00');
INSERT INTO "epics_channel" VALUES(25, 'TBDB002L03', 'buncher_on_off', 0.0, 0.0, 'ON', '2017-01-27 09:22:04');
INSERT INTO "epics_channel" VALUES(26, 'TDDB001E04', 'buncher_amp', 34.9, 0.0, 'NA', '2017-03-31 12:27:38');
INSERT INTO "epics_channel" VALUES(27, 'TDDB001E02', 'buncher_ph', 227.2, 0.0, 'NA', '2016-03-29 10:42:00');
INSERT INTO "epics_channel" VALUES(28, 'TDDB001L03', 'buncher_on_off', 0.0, 0.0, 'ON', '2016-06-21 10:01:45');
INSERT INTO "epics_channel" VALUES(29, 'TBDB001E01', 'buncher_amp', 0.0, 0.0, 'NA', '2017-01-27 09:21:52');
INSERT INTO "epics_channel" VALUES(30, 'TBDB001E02', 'buncher_ph', 0.0, 0.0, 'NA', '2016-05-26 08:32:38');
INSERT INTO "epics_channel" VALUES(31, 'TBDB001L03', 'buncher_on_off', 1.0, 0.0, 'OFF', '2017-01-27 09:21:43');
CREATE TABLE quad_family(
  id integer primary key,
  name text,
  l_eff_cal double precision check(l_eff_cal > 0.0),-- effective magnet length
  a0_cal double precision not null default 0.0,
  a1_cal double precision not null default 0.0,
  a2_cal double precision not null default 0.0,
  a3_cal double precision not null default 0.0,
  a4_cal double precision not null default 0.0,
  a6_cal double precision not null default 0.0,
  a8_cal double precision not null default 0.0,
  unique(name, l_eff_cal, a0_cal, a1_cal, a2_cal, a3_cal, a4_cal, a6_cal, a8_cal)
);
INSERT INTO "quad_family" VALUES(1, 'TBQL01V1', 1.0, 0.0, 14.772, -0.008595, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(2, 'TBQL01V2', 1.0, 0.0, 14.647, -0.00617, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(3, 'TBQL01V3', 1.0, 0.0, 14.423, -0.002663, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(4, 'TBQL02V1', 1.0, 0.0, 14.888, -0.008325, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(5, 'TBQL02V2', 1.0, 0.0, 14.857, -0.00855, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(6, 'TBQL02V3', 1.0, 0.0, 14.831, -0.00802, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(7, 'TBQL03V1', 1.0, 0.0, 14.831, -0.00802, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(8, 'TBQL03V2', 1.0, 0.0, 14.831, -0.00802, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(9, 'TBQL04V1', 1.0, 0.0, 8.9151, -0.003034, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(10, 'TBQL04V2', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(11, 'TBQL05V1', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(12, 'TBQL05V2', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(13, 'TBQL05V3', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(14, 'TBQL05V4', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(15, 'TBQL06V1', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(16, 'TBQL06V2', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(17, 'TBQL06V3', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(18, 'TBQL06V4', 1.0, 0.0, 8.9151, -0.003024, 0.0, 0.0, 0.0, 0.0);
INSERT INTO "quad_family" VALUES(19, 'TDQL01V1', 1.0, 2.7, 8.0285, 0.000576, 0.0, 0.0, -8.294e-12, 0.0);
INSERT INTO "quad_family" VALUES(20, 'TDQL01V2', 1.0, 2.6, 8.0414, 0.000662, 0.0, 0.0, -1.0271e-11, 0.0);
INSERT INTO "quad_family" VALUES(21, 'TDQL01V3', 1.0, 2.6, 8.0419, 0.000549, 0.0, 0.0, -9.184e-12, 0.0);
INSERT INTO "quad_family" VALUES(22, 'TDQL01V4', 1.0, 2.7, 8.093, -7.2e-05, 0.0, 0.0, -4.224e-12, 0.0);
CREATE TABLE diagnostics(
  id integer primary key,
  name text,
  view_index double precision unique,
  diag_type text,
  monitor integer not null default 0,
  model_index integer unique,
  model_type text not null default 'diagnostics'
);
INSERT INTO "diagnostics" VALUES(1, 'TBEM01', 9.0, 'EM', 0, 11, 'diagnostics');
INSERT INTO "diagnostics" VALUES(2, 'TBEM01COL', 17.0, 'HP', 0, 20, 'diagnostics');
INSERT INTO "diagnostics" VALUES(3, 'TBEM02', 29.0, 'EM', 0, 34, 'diagnostics');
INSERT INTO "diagnostics" VALUES(4, 'TBEM02COL', 35.0, 'HP', 0, 41, 'diagnostics');
INSERT INTO "diagnostics" VALUES(7, 'TBEM03', 46.0, 'EM', 0, 54, 'diagnostics');
INSERT INTO "diagnostics" VALUES(8, 'TBEM03COL', 56.0, 'HP', 0, 69, 'diagnostics');
INSERT INTO "diagnostics" VALUES(9, 'TBEM04', 70.0, 'EM', 0, 85, 'diagnostics');
INSERT INTO "diagnostics" VALUES(10, 'TBEM04COL', 76.0, 'HP', 0, 92, 'diagnostics');
INSERT INTO "diagnostics" VALUES(11, 'TDEM01', 96.0, 'EM', 0, 116, 'diagnostics');
INSERT INTO "diagnostics" VALUES(12, 'TDEM01COL', 106.0, 'HP', 0, 127, 'diagnostics');
CREATE TABLE drift(
  id integer primary key,
  name text unique,
  view_index double precision unique,
  length_model double precision,
  aperture_model double precision,
  model_index integer unique,
  model_type text not null default 'drift'
);
INSERT INTO "drift" VALUES(1, 'TBDR01', 1.0, 0.0524, 0.0254, 1, 'drift');
INSERT INTO "drift" VALUES(2, 'TBDR02', 3.0, 0.05695, 0.0254, 3, 'drift');
INSERT INTO "drift" VALUES(3, 'TBDR03', 5.0, 0.05695, 0.0254, 5, 'drift');
INSERT INTO "drift" VALUES(4, 'TBDR04', 7.0, 0.11621, 0.0254, 7, 'drift');
INSERT INTO "drift" VALUES(5, 'TBDR05', 8.0, 0.08201, 0.0254, 10, 'drift');
INSERT INTO "drift" VALUES(7, 'TBDR06', 10.0, 0.05903, 0.0254, 13, 'drift');
INSERT INTO "drift" VALUES(8, 'TBDR07', 12.0, 0.07273, 0.0254, 15, 'drift');
INSERT INTO "drift" VALUES(9, 'TBDR08', 14.0, 0.07273, 0.0254, 17, 'drift');
INSERT INTO "drift" VALUES(10, 'TBDR09', 16.0, 0.0421, 0.0254, 19, 'drift');
INSERT INTO "drift" VALUES(12, 'TBDR10', 18.0, 0.06253, 0.0254, 21, 'drift');
INSERT INTO "drift" VALUES(13, 'TBDR11', 20.0, 0.46278, 0.0254, 23, 'drift');
INSERT INTO "drift" VALUES(14, 'TBDR12', 21.0, 0.05575, 0.0254, 26, 'drift');
INSERT INTO "drift" VALUES(15, 'TBDR13', 22.0, 0.025, 0.0254, 27, 'drift');
INSERT INTO "drift" VALUES(16, 'TBDR14', 23.0, 0.025, 0.0254, 28, 'drift');
INSERT INTO "drift" VALUES(17, 'TBDR15', 24.0, 0.025, 0.0254, 29, 'drift');
INSERT INTO "drift" VALUES(18, 'TBDR16', 25.0, 0.025, 0.0254, 30, 'drift');
INSERT INTO "drift" VALUES(19, 'TBDR17', 26.0, 0.51985, 0.0254, 31, 'drift');
INSERT INTO "drift" VALUES(20, 'TBDR18', 28.0, 0.05868, 0.0254, 33, 'drift');
INSERT INTO "drift" VALUES(22, 'TBDR19', 30.0, 0.04748, 0.0254, 36, 'drift');
INSERT INTO "drift" VALUES(23, 'TBDR20', 32.0, 0.07273, 0.0254, 38, 'drift');
INSERT INTO "drift" VALUES(24, 'TBDR21', 34.0, 0.0421, 0.0254, 40, 'drift');
INSERT INTO "drift" VALUES(26, 'TBDR22', 36.0, 0.08471, 0.0254, 42, 'drift');
INSERT INTO "drift" VALUES(27, 'TBDR23', 37.0, 0.1853, 0.0254, 45, 'drift');
INSERT INTO "drift" VALUES(28, 'TBDR24', 39.0, 0.19858, 0.0254, 47, 'drift');
INSERT INTO "drift" VALUES(29, 'TBDR25', 41.0, 0.06018, 0.0254, 49, 'drift');
INSERT INTO "drift" VALUES(30, 'TBDR26', 43.0, 0.07273, 0.0254, 51, 'drift');
INSERT INTO "drift" VALUES(31, 'TBDR27', 45.0, 0.07288, 0.0254, 53, 'drift');
INSERT INTO "drift" VALUES(33, 'TBDR28', 47.0, 0.06861, 0.0254, 56, 'drift');
INSERT INTO "drift" VALUES(34, 'TBDR29', 48.0, 0.3493, 0.0254, 59, 'drift');
INSERT INTO "drift" VALUES(35, 'TBDR30', 50.0, 0.4082, 0.0254, 61, 'drift');
INSERT INTO "drift" VALUES(36, 'TBDR31', 52.0, 0.20557, 0.0254, 63, 'drift');
INSERT INTO "drift" VALUES(38, 'TBDR32', 54.0, 0.15208, 0.0254, 65, 'drift');
INSERT INTO "drift" VALUES(39, 'TBDR33', 55.0, 0.07395, 0.0254, 68, 'drift');
INSERT INTO "drift" VALUES(41, 'TBDR34', 57.0, 0.05286, 0.0254, 70, 'drift');
INSERT INTO "drift" VALUES(42, 'TBDR35', 59.0, 0.07273, 0.0254, 72, 'drift');
INSERT INTO "drift" VALUES(43, 'TBDR36', 61.0, 0.07679, 0.0254, 74, 'drift');
INSERT INTO "drift" VALUES(44, 'TBDR37', 63.0, 0.07679, 0.0254, 76, 'drift');
INSERT INTO "drift" VALUES(45, 'TBDR38', 65.0, 0.06018, 0.0254, 78, 'drift');
INSERT INTO "drift" VALUES(46, 'TBDR39', 67.0, 0.79308, 0.0254, 80, 'drift');
INSERT INTO "drift" VALUES(47, 'TBDR40', 68.0, 0.79308, 0.0254, 81, 'drift');
INSERT INTO "drift" VALUES(48, 'TBDR41', 69.0, 0.14283, 0.0254, 84, 'drift');
INSERT INTO "drift" VALUES(50, 'TBDR42', 71.0, 0.04748, 0.0254, 87, 'drift');
INSERT INTO "drift" VALUES(51, 'TBDR43', 73.0, 0.07679, 0.0254, 89, 'drift');
INSERT INTO "drift" VALUES(52, 'TBDR44', 75.0, 0.0421, 0.0254, 91, 'drift');
INSERT INTO "drift" VALUES(54, 'TBDR45', 77.0, 0.08793, 0.0254, 93, 'drift');
INSERT INTO "drift" VALUES(55, 'TBDR46', 79.0, 0.06018, 0.0254, 95, 'drift');
INSERT INTO "drift" VALUES(56, 'TBDR47', 81.0, 0.07273, 0.0254, 97, 'drift');
INSERT INTO "drift" VALUES(57, 'TBDR48', 83.0, 0.14425, 0.0254, 99, 'drift');
INSERT INTO "drift" VALUES(58, 'TBDR49', 84.0, 0.29965, 0.0254, 102, 'drift');
INSERT INTO "drift" VALUES(59, 'TDDR01', 86.0, 0.15835, 0.0254, 104, 'drift');
INSERT INTO "drift" VALUES(61, 'TDDR02', 88.0, 0.30645, 0.0254, 106, 'drift');
INSERT INTO "drift" VALUES(62, 'TDDR03', 89.0, 0.02, 0.0254, 107, 'drift');
INSERT INTO "drift" VALUES(63, 'TDDR04', 90.0, 0.02, 0.0254, 108, 'drift');
INSERT INTO "drift" VALUES(64, 'TDDR05', 92.0, 0.02, 0.0254, 110, 'drift');
INSERT INTO "drift" VALUES(65, 'TDDR06', 93.0, 0.02, 0.0254, 111, 'drift');
INSERT INTO "drift" VALUES(66, 'TDDR07', 94.0, 0.36877, 0.0254, 112, 'drift');
INSERT INTO "drift" VALUES(67, 'TDDR08', 95.0, 0.11462, 0.0254, 115, 'drift');
INSERT INTO "drift" VALUES(69, 'TDDR09', 97.0, 0.08405, 0.0254, 118, 'drift');
INSERT INTO "drift" VALUES(70, 'TDDR10', 99.0, 0.07717, 0.0254, 120, 'drift');
INSERT INTO "drift" VALUES(71, 'TDDR11', 101.0, 0.19756, 0.0254, 122, 'drift');
INSERT INTO "drift" VALUES(72, 'TDDR12', 103.0, 0.07846, 0.0254, 124, 'drift');
INSERT INTO "drift" VALUES(73, 'TDDR13', 105.0, 0.04219, 0.0254, 126, 'drift');
INSERT INTO "drift" VALUES(75, 'TDDR14', 107.0, 0.35632, 0.01, 128, 'drift');
CREATE TABLE dipole(
  id integer primary key,
  name text unique,
  view_index double precision unique,
  rho_model double precision,
  angle_model double precision,
  half_gap_model double precision,
  edge_angle1_model double precision not null default 0.0,
  edge_angle2_model double precision not null default 0.0,
  k1_model double precision not null default 0.45,
  k2_model double precision not null default 2.8,
  field_index_model double precision not null default 0.0,
  kenergy_model double precision default 0.75,
  channel integer references epics_channel(id)
          on delete restrict,
  model_index integer unique,
  model_type text not null default 'dipole'
);
INSERT INTO "dipole" VALUES(1, 'TBBM01', 38.0, 0.56634, 1.41371669411541, 0.0381, 0.445058959258554, 0.445058959258554, 0.45, 2.8, 0.0, 0.75, NULL, 46, 'dipole');
INSERT INTO "dipole" VALUES(2, 'TDBM01', 85.0, 1.6372, 0.15707963267949, 0.03963, 0.150237942011672, 0.0, 0.243, 2.801, 0.0, 0.75, NULL, 103, 'dipole');
CREATE TABLE steerer(
  id integer primary key,
  name text unique,
  view_index double precision unique,
  bl_h_model double precision not null default 0.0,
  bl_v_model double precision not null default 0.0,
  channel integer references epics_channel(id)
          on delete restrict,
  model_index integer unique,
  model_type text not null default 'steerer'
);
INSERT INTO "steerer" VALUES(1, 'TBSM01X', 7.6, 0.0, 0.0, NULL, 8, 'steerer');
INSERT INTO "steerer" VALUES(2, 'TBSM01Y', 7.7, 0.0, 0.0, NULL, 9, 'steerer');
INSERT INTO "steerer" VALUES(3, 'TBSM02X', 20.6, 0.0, 0.0, NULL, 24, 'steerer');
INSERT INTO "steerer" VALUES(4, 'TBSM02Y', 20.7, 0.0, 0.0, NULL, 25, 'steerer');
INSERT INTO "steerer" VALUES(5, 'TBSM03X', 36.6, 0.0, 0.0, NULL, 43, 'steerer');
INSERT INTO "steerer" VALUES(6, 'TBSM03Y', 36.7, 0.0, 0.0, NULL, 44, 'steerer');
INSERT INTO "steerer" VALUES(7, 'TBSM04X', 47.6, 0.0, 0.0, NULL, 57, 'steerer');
INSERT INTO "steerer" VALUES(8, 'TBSM04Y', 47.7, 0.0, 0.0, NULL, 58, 'steerer');
INSERT INTO "steerer" VALUES(9, 'TBSM05X', 54.6, 0.0, 0.0, NULL, 66, 'steerer');
INSERT INTO "steerer" VALUES(10, 'TBSM05Y', 54.7, 0.0, 0.0, NULL, 67, 'steerer');
INSERT INTO "steerer" VALUES(11, 'TBSM06X', 68.6, 0.0, 0.0, NULL, 82, 'steerer');
INSERT INTO "steerer" VALUES(12, 'TBSM06Y', 68.7, 0.0, 0.0, NULL, 83, 'steerer');
INSERT INTO "steerer" VALUES(13, 'TBSM07X', 83.6, 0.0, 0.0, '', 100, 'steerer');
INSERT INTO "steerer" VALUES(14, 'TBSM07Y', 83.7, 0.0, 0.0, '', 101, 'steerer');
INSERT INTO "steerer" VALUES(15, 'TDSM01X', 94.6, 0.0, 0.0, NULL, 113, 'steerer');
INSERT INTO "steerer" VALUES(16, 'TDSM01Y', 94.7, 0.0, 0.0, NULL, 114, 'steerer');
CREATE TABLE quad(
  id integer primary key,
  name text unique,
  view_index double precision unique,
  monitor integer not null default 0,
  gradient_model double precision default 0.0,
  length_model double precision,
  aperture_model double precision,
  family_cal  integer references quad_family(id)
          on delete restrict,
  shunt_cal double precision,
  polarity_design integer not null default 1.0,
  channel integer references epics_channel(id)
          on delete restrict,
  model_index integer unique,
  model_type text not null default 'quad'
);
INSERT INTO "quad" VALUES(1, 'TBQL01V1', 2.0, 0, 1.2116696080242, 0.1022, 0.0254, 1, 0.0, 1, 1, 2, 'quad');
INSERT INTO "quad" VALUES(2, 'TBQL01V2', 4.0, 0, -3.6325072643175, 0.1022, 0.0254, 2, 0.0, -1, 2, 4, 'quad');
INSERT INTO "quad" VALUES(3, 'TBQL01V3', 6.0, 0, 2.96055295675425, 0.1022, 0.0254, 3, 0.0, 1, 3, 6, 'quad');
INSERT INTO "quad" VALUES(4, 'TBQL02V1', 11.0, 0, -3.669389878668, 0.1034, 0.0254, 4, 0.0, -1, 4, 14, 'quad');
INSERT INTO "quad" VALUES(5, 'TBQL02V2', 13.0, 0, 5.360978402762, 0.1034, 0.0254, 5, 0.0, 1, 5, 16, 'quad');
INSERT INTO "quad" VALUES(6, 'TBQL02V3', 15.0, 0, -2.8841954214528, 0.1034, 0.0254, 6, 0.0, -1, 6, 18, 'quad');
INSERT INTO "quad" VALUES(7, 'TBQL03V1', 31.0, 0, -0.4450951699128, 0.1034, 0.0254, 7, 0.0, -1, 7, 37, 'quad');
INSERT INTO "quad" VALUES(8, 'TBQL03V2', 33.0, 0, 0.2436048805728, 0.1034, 0.0254, 8, 0.0, 1, 8, 39, 'quad');
INSERT INTO "quad" VALUES(9, 'TBQL04V1', 42.0, 0, -0.05722243694424, 0.1034, 0.0254, 9, 0.0, -1, 9, 50, 'quad');
INSERT INTO "quad" VALUES(10, 'TBQL04V2', 44.0, 0, 0.0, 0.1034, 0.0254, 10, 0.0, 1, 10, 52, 'quad');
INSERT INTO "quad" VALUES(11, 'TBQL05V1', 58.0, 0, 0.208002709716, 0.1034, 0.0254, 11, 0.0, 1, 11, 71, 'quad');
INSERT INTO "quad" VALUES(12, 'TBQL05V2', 60.0, 0, -2.92456430035584, 0.1034, 0.0254, 12, 0.0, -1, 12, 73, 'quad');
INSERT INTO "quad" VALUES(13, 'TBQL05V3', 62.0, 0, 4.85421897275136, 0.1034, 0.0254, 13, 0.0, 1, 13, 75, 'quad');
INSERT INTO "quad" VALUES(14, 'TBQL05V4', 64.0, 0, -2.45354012550864, 0.1034, 0.0254, 14, 0.0, -1, 14, 77, 'quad');
INSERT INTO "quad" VALUES(15, 'TBQL06V1', 72.0, 0, 0.878745552096, 0.1034, 0.0254, 15, 0.0, 1, 15, 88, 'quad');
INSERT INTO "quad" VALUES(16, 'TBQL06V2', 74.0, 0, -2.98868619386496, 0.1034, 0.0254, 16, 0.0, -1, 16, 90, 'quad');
INSERT INTO "quad" VALUES(17, 'TBQL06V3', 80.0, 0, 3.79220538715536, 0.1034, 0.0254, 17, 0.0, 1, 17, 96, 'quad');
INSERT INTO "quad" VALUES(18, 'TBQL06V4', 82.0, 0, -1.833786641556, 0.1034, 0.0254, 18, 0.0, -1, 18, 98, 'quad');
INSERT INTO "quad" VALUES(19, 'TDQL01V1', 98.0, 0, -2.50565511698528, 0.1028, 0.0254, 19, 0.0, -1, 19, 119, 'quad');
INSERT INTO "quad" VALUES(20, 'TDQL01V2', 100.0, 0, 4.65965900699259, 0.10278, 0.0254, 20, 0.0, 1, 20, 121, 'quad');
INSERT INTO "quad" VALUES(21, 'TDQL01V3', 102.0, 0, -5.94156896619775, 0.1028, 0.0254, 21, 0.0, -1, 21, 123, 'quad');
INSERT INTO "quad" VALUES(22, 'TDQL01V4', 104.0, 0, 5.27735624624285, 0.1028, 0.0254, 22, 0.0, 1, 22, 125, 'quad');
CREATE TABLE rotation(
  id integer primary key,
  name text,
  view_index double precision unique,
  angle_model double precision not null default 0.0,
  model_index integer unique, 
  model_type text not null default 'rotation'
);
CREATE TABLE spch_comp(
  id integer primary key,
  name text,
  view_index double precision unique,
  fraction_model double precision not null default 1.0,
  model_index integer unique, 
  model_type text not null default 'spch_comp'
);
INSERT INTO "spch_comp" VALUES(1, 'spch_comp_1', 0.5, 0.6, 0, 'spch_comp');
INSERT INTO "spch_comp" VALUES(2, 'spch_comp_2', 9.2, 0.74, 12, 'spch_comp');
INSERT INTO "spch_comp" VALUES(3, 'spch_comp_3', 29.2, 0.15, 35, 'spch_comp');
INSERT INTO "spch_comp" VALUES(4, 'spch_comp_4', 46.2, 0.26, 55, 'spch_comp');
INSERT INTO "spch_comp" VALUES(5, 'spch_comp_5', 70.2, 0.4, 86, 'spch_comp');
INSERT INTO "spch_comp" VALUES(6, 'spch_comp_6', 96.2, 0.4, 117, 'spch_comp');
INSERT INTO "spch_comp" VALUES(7, 'spch_comp_7', 109.0, 1.0, 129, 'spch_comp');
CREATE TABLE caperture(
  id integer primary key,
  name text,
  view_index double precision unique,
  aperture_model double precision not null default 0.0,
  in_out_model integer not null default 1,
  model_index integer unique,
  model_type text not null default 'caperture'
);
INSERT INTO "caperture" VALUES(1, 'TBBA01', 19.0, 0.0254, 0, 22, 'caperture');
INSERT INTO "caperture" VALUES(2, 'TBBA02', 27.0, 0.0254, 0, 32, 'caperture');
INSERT INTO "caperture" VALUES(3, 'TBBA04', 53.0, 0.01, 1, 64, 'caperture');
INSERT INTO "caperture" VALUES(4, 'TDBA01', 87.0, 0.01, 1, 105, 'caperture');
CREATE TABLE raperture(
  id integer primary key,
  name text,
  view_index double precision unique,
  aperture_xl_model double precision not null default 0.0,
  aperture_xr_model double precision not null default 0.0,
  aperture_yt_model double precision not null default 0.0,
  aperture_yb_model double precision not null default 0.0,
  in_out_model integer not null default 1,
  model_index integer unique, 
  model_type text not null default 'raperture'
);
INSERT INTO "raperture" VALUES(1, 'TBFJ01', 40.0, 0.0, 0.0, 0.0, 0.0, 0, 48, 'raperture');
INSERT INTO "raperture" VALUES(2, 'TBFJ02', 66.0, 0.0, 0.0, 0.0, 0.0, 0, 79, 'raperture');
INSERT INTO "raperture" VALUES(3, 'TBFJ03', 78.0, 0.00147477110418, 0.00147477110418, 0.00431303729257, 0.00431303729257, 0, 94, 'raperture');
CREATE TABLE buncher(
  id integer primary key,
  name text, 
  view_index double precision unique,
  on_off integer not null default 1,
  phase_model double precision not null default 0.0,
  phase_offset_cal double precision not null default 0.0,
  voltage_model double precision not null default 0.0,
  c0_cal double precision not null default 1.0,
  c1_cal double precision not null default 0.0,
  c2_cal double precision not null default 0.0,
  c3_cal double precision not null default 0.0,
  c4_cal double precision not null default 0.0,
  frequency_model double precision not null default 0.0,
  aperture_model double precision not null default 0.0,
  amplitude_channel integer references epics_channel(id)
          on delete restrict,
  phase_channel integer references epics_channel(id)
          on delete restrict,
  on_off_channel integer references epics_channel(id)
          on delete restrict, 
  model_index integer unique, 
  model_type text not null default 'buncher'
);
INSERT INTO "buncher" VALUES(1, 'TBDB01', 49.0, 0, -1.5707963267949, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 201.25, 0.01, 29, 30, 31, 60, 'buncher');
INSERT INTO "buncher" VALUES(2, 'TBDB02', 51.0, 1, 2.96705972839036, -100.0, 0.00398831622421552, 1.07190247, 0.07971817, -0.00032347, 3.7308e-06, -1.5969e-08, 201.25, 0.01, 23, 24, 25, 62, 'buncher');
INSERT INTO "buncher" VALUES(3, 'TDDB01', 91.0, 1, 5.52313456050885, 89.2523, 0.00965338223330795, 2.76867036, 0.22313134, -0.0010578, 1.0405e-05, -3.8059e-08, 201.25, 0.01, 26, 27, 28, 109, 'buncher');
CREATE VIEW linac as 
  select view_index, name, model_type, model_index from buncher union
  select view_index, name, model_type, model_index from quad union
  select view_index, name, model_type, model_index from dipole union
  select view_index, name, model_type, model_index from drift union
  select view_index, name, model_type, model_index from rotation union
  select view_index, name, model_type, model_index from caperture union
  select view_index, name, model_type, model_index from raperture union
  select view_index, name, model_type, model_index from diagnostics union
  select view_index, name, model_type, model_index from steerer union
  select view_index, name, model_type, model_index from spch_comp;
CREATE VIEW channel_list as 
  select view_index, name, model_type, (select lcs_name from epics_channel where id = amplitude_channel) as channel1, 
    (select lcs_name from epics_channel where id = phase_channel) as channel2, 
    (select lcs_name from epics_channel where id = on_off_channel) as channel3,
    (select lcs_name from epics_channel where id = NULL) as channel4 from buncher union
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), NULL, NULL, NULL from quad union
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), NULL, NULL, NULL from steerer union
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), NULL, NULL, NULL from dipole;
CREATE TRIGGER insert_epics_channel after insert on epics_channel
begin
  update epics_channel set update_time = datetime('now', 'localtime') where rowid = new.rowid;
end;
CREATE TRIGGER update_epics_channel after update of value on epics_channel
begin
  update epics_channel set update_time = datetime('now', 'localtime') where rowid = new.rowid;
  update quad set 
    polarity_design = (select case when value >= 0.0 then 1.0 else -1.0 end from epics_channel where quad.channel=id)
    where exists(select * from epics_channel where quad.channel=new.rowid AND value_type = 'DVM');
  update quad set 
    gradient_model= 0.01*polarity_design*(((select a0_cal from quad_family mf where quad.family_cal = mf.id)+
              (select a1_cal from quad_family mf where quad.family_cal = mf.id)*
              ((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal)+
              (select a2_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 2.0)+
              (select a3_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 3.0)+
              (select a4_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 4.0)+
              (select a6_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 6.0)+
              (select a8_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 8.0))/
              (select l_eff_cal from quad_family mf where quad.family_cal = mf.id))
    where exists(select * from epics_channel where quad.channel=new.rowid);
---- update buncher amplitude ----
  update buncher set
    voltage_model = on_off*0.001*(c0_cal + c1_cal* (select value from epics_channel where buncher.amplitude_channel=id) +
                    c2_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 2.0) + 
                    c3_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 3.0) + 
                    c4_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 4.0))
    where exists(select * from epics_channel where buncher.amplitude_channel=new.rowid);
---- update buncher phase ----
  update buncher set
    phase_model = (phase_offset_cal + (select value from epics_channel where 
      buncher.phase_channel=id))*pi()/180.0
    where exists(select * from epics_channel where buncher.phase_channel=new.rowid);
---- update buncher on_off ----
  update buncher set
    on_off = (select case when (select value from epics_channel where buncher.on_off_channel = id) = 0 then 1 else 0 end)
    where exists(select * from epics_channel where buncher.on_off_channel=new.rowid);
  update epics_channel set
    value_txt = (select case when value = 1 then 'OFF' else 'ON' end) where rowid = new.rowid and value_type = 'buncher_on_off';
end;
CREATE TRIGGER update_buncher_on_off AFTER UPDATE OF on_off ON buncher
begin
  update buncher set
    voltage_model = on_off*0.001*(c0_cal + c1_cal* (select value from epics_channel where buncher.amplitude_channel=id) +
                    c2_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 2.0) + 
                    c3_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 3.0) + 
                    c4_cal* power((select value from epics_channel where buncher.amplitude_channel=id), 4.0))
    where rowid = new.rowid;
end;
CREATE TRIGGER update_buncher_phase_offset_cal AFTER UPDATE OF phase_offset_cal ON buncher 
BEGIN
update buncher set phase_model = (phase_offset_cal + (select value from epics_channel where buncher.phase_channel = epics_channel.id))*pi()/180.0 where rowid = new.rowid;
END;
COMMIT;
