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
INSERT INTO "epics_channel" VALUES(1, 'TRMP001V04', 'dipole', 44.094, 0.0, 'NA', '2014-08-28 15:07:17');
INSERT INTO "epics_channel" VALUES(2, 'TRMP002V04', 'dipole', 40.0, 0.0, 'NA', '2014-11-24 11:26:16');
INSERT INTO "epics_channel" VALUES(3, 'TRQM005V01', 'DVM', 0.0, 0.05, 'NA', '2014-08-25 13:47:35');
INSERT INTO "epics_channel" VALUES(4, 'TRQM006V01', 'DVM', 0.0, 0.05, 'NA', '2014-08-25 12:10:40');
INSERT INTO "epics_channel" VALUES(5, 'TRQM007V01', 'DVM', 0.0, 0.05, 'NA', '2014-08-25 12:10:56');
INSERT INTO "epics_channel" VALUES(6, 'TRQM008V01', 'DVM', 0.0, 0.05, 'NA', '2014-08-25 12:11:15');
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
  a14_cal double precision not null default 0.0,
  unique(name, l_eff_cal, a0_cal, a1_cal, a2_cal, a3_cal, a4_cal, a6_cal, a14_cal)
);
INSERT INTO "quad_family" VALUES(1, 'TRQM005V01', 1.0, 0.0, 31.344, -0.00458, 0.0, 0.0, 0.0, -7.33e-27);
INSERT INTO "quad_family" VALUES(2, 'TRQM006V01', 1.0, 0.0, 31.064, -0.00464, 0.0, 0.0, 0.0, -6.65e-27);
INSERT INTO "quad_family" VALUES(3, 'TRQM007V01', 1.0, 0.0, 31.163, -0.0046, 0.0, 0.0, 0.0, -6.61e-27);
INSERT INTO "quad_family" VALUES(4, 'TRQM008V01', 1.0, 0.0, 31.289, -0.00469, 0.0, 0.0, 0.0, -7.03e-27);
CREATE TABLE diagnostics(
  id integer primary key,
  name text,
  view_index double precision unique,
  diag_type text,
  monitor integer not null default 0,
  model_index integer unique,
  model_type text not null default 'diagnostics'
);
INSERT INTO "diagnostics" VALUES(1, 'TREM01', 0.3, 'EM', 0, 480, 'diagnostics');
INSERT INTO "diagnostics" VALUES(2, 'TRHP06', 6.8, 'HP', 0, 488, 'diagnostics');
INSERT INTO "diagnostics" VALUES(3, 'TRHP07', 15.0, 'HP', 0, 503, 'diagnostics');
INSERT INTO "diagnostics" VALUES(4, 'TRHP08', 23.2, 'HP', 0, 517, 'diagnostics');
INSERT INTO "diagnostics" VALUES(5, 'TREM02', 31.0, 'EM', 0, 525, 'diagnostics');
CREATE TABLE drift(
  id integer primary key,
  name text unique,
  view_index double precision unique,
  length_model double precision,
  aperture_model double precision,
  dz_design double precision default 0.0,
  channel integer references epics_channel(id)
          on delete restrict,
  channel2 integer references epics_channel(id)
          on delete restrict,
  model_index integer unique,
  model_type text not null default 'drift'
);
INSERT INTO "drift" VALUES(1, 'TRDR01', 0.0, 0.1751, 0.0254, 0.0, '', '', 479, 'drift');
INSERT INTO "drift" VALUES(2, 'TRDR02', 0.5, 0.0575, 0.0254, 0.0, '', '', 481, 'drift');
INSERT INTO "drift" VALUES(3, 'TRDR03', 2.0, 0.1952, 0.0254, 0.0, '', '', 483, 'drift');
INSERT INTO "drift" VALUES(4, 'TRDR04', 4.0, 0.141, 0.0254, 0.0, '', '', 485, 'drift');
INSERT INTO "drift" VALUES(5, 'TRDR05', 6.0, 0.1321, 0.0254, 0.0, '', '', 487, 'drift');
INSERT INTO "drift" VALUES(6, 'TRDR06', 8.0, 0.0737, 0.0254, 0.0, '', '', 490, 'drift');
INSERT INTO "drift" VALUES(7, 'TRDR07', 10.0, 0.087, 0.0254, 0.0, '', '', 492, 'drift');
INSERT INTO "drift" VALUES(8, 'TRDR08', 12.0, 0.0437705652838511, 0.0254, 0.04137722, 1, 2, 497, 'drift');
INSERT INTO "drift" VALUES(9, 'TRDR09', 14.0, 0.05, 0.0254, 0.0, '', '', 502, 'drift');
INSERT INTO "drift" VALUES(10, 'TRDR10', 16.0, 0.05, 0.0254, 0.0, '', '', 504, 'drift');
INSERT INTO "drift" VALUES(11, 'TRDR11', 18.0, 0.0437705652838511, 0.0254, 0.04137722, 1, 2, 509, 'drift');
INSERT INTO "drift" VALUES(12, 'TRDR12', 20.0, 0.09, 0.0254, 0.0, '', '', 513, 'drift');
INSERT INTO "drift" VALUES(13, 'TRDR13', 22.0, 0.0737, 0.0254, 0.0, '', '', 515, 'drift');
INSERT INTO "drift" VALUES(14, 'TRDR14', 24.0, 0.1291, 0.0254, 0.0, '', '', 518, 'drift');
INSERT INTO "drift" VALUES(15, 'TRDR15', 26.0, 0.141, 0.0254, 0.0, '', '', 520, 'drift');
INSERT INTO "drift" VALUES(16, 'TRDR16', 28.0, 0.1952, 0.0254, 0.0, '', '', 522, 'drift');
INSERT INTO "drift" VALUES(17, 'TRDR17', 30.0, 0.0575, 0.0254, 0.0, '', '', 524, 'drift');
INSERT INTO "drift" VALUES(18, 'TRDR18', 32.0, 0.1284, 0.0254, 0.0, '', '', 526, 'drift');
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
  kenergy_model double precision default 100.0,
  bfield_ratio_tmp double precision default 1.0,
  bfield_ratio_cal double precision default 1.0,
  current_ratio_tmp double precision default 0.0,
  kenergy_design double precision default 100.0,
  mc2_design double precision default 939.272,
  channel integer references epics_channel(id)
          on delete restrict,
  channel2 integer references epics_channel(id)
          on delete restrict,
  model_index integer unique,
  model_type text not null default 'dipole'
);
INSERT INTO "dipole" VALUES(1, 'TRBM01', 1.0, 1.32155958887372, 0.349065850398866, 0.02853, 0.0, 0.349065850398866, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, NULL, 100.0, 939.272, 1, '', 482, 'dipole');
INSERT INTO "dipole" VALUES(2, 'TRBM05', 11.0, 28.46743942931, 0.016845943445215, 0.02853, 0.349065850398866, -0.332219906953651, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, 0.9071529006214, 100.0, 939.272, 1, 2, 494, 'dipole');
INSERT INTO "dipole" VALUES(3, 'TRBM06', 13.0, 1.38589788835821, 0.332219906953651, 0.02853, 0.332219906953651, 0.0, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, 0.9071529006214, 100.0, 939.272, 1, 2, 499, 'dipole');
INSERT INTO "dipole" VALUES(4, 'TRBM07', 17.0, 1.38589788835821, 0.332219906953651, 0.02853, 0.0, 0.332219906953651, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, 0.9071529006214, 100.0, 939.272, 1, 2, 506, 'dipole');
INSERT INTO "dipole" VALUES(5, 'TRBM08', 19.0, 28.46743942931, 0.016845943445215, 0.02853, -0.332219906953651, 0.349065850398866, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, 0.9071529006214, 100.0, 939.272, 1, 2, 511, 'dipole');
INSERT INTO "dipole" VALUES(6, 'TRBM04', 29.0, 1.32155958887372, 0.349065850398866, 0.02853, 0.349065850398866, 0.0, -0.424, 0.002, -0.204, 100.0, 1.0, 44.094, NULL, 100.0, 939.272, 1, '', 523, 'dipole');
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
INSERT INTO "steerer" VALUES(1, 'TRSM05Y', 3.0, 0.0, 0.0, NULL, 484, 'steerer');
INSERT INTO "steerer" VALUES(2, 'TRSM06X', 5.0, 0.0, 0.0, NULL, 486, 'steerer');
INSERT INTO "steerer" VALUES(3, 'TRSM07X', 25.0, 0.0, 0.0, NULL, 519, 'steerer');
INSERT INTO "steerer" VALUES(4, 'TRSM08Y', 27.0, 0.0, 0.0, NULL, 521, 'steerer');
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
INSERT INTO "quad" VALUES(1, 'TRQM05', 7.0, 0, 16.341, 0.12, 0.0, 1, 0.0, 1, 3, 489, 'quad');
INSERT INTO "quad" VALUES(2, 'TRQM06', 9.0, 0, -12.689, 0.12, 0.0, 2, 0.0, -1, 4, 491, 'quad');
INSERT INTO "quad" VALUES(3, 'TRQM07', 21.0, 0, -19.648, 0.12, 0.0, 3, 0.0, -1, 5, 514, 'quad');
INSERT INTO "quad" VALUES(4, 'TRQM08', 23.0, 0, 20.7282, 0.12, 0.0, 4, 0.0, 1, 6, 516, 'quad');
CREATE TABLE rotation(
  id integer primary key,
  name text,
  view_index double precision unique,
  angle_model double precision not null default 0.0,
  model_index integer unique, 
  model_type text not null default 'rotation'
);
INSERT INTO "rotation" VALUES(1, 'TRBM05-BEFORE', 10.5, 3.14159265358979, 493, 'rotation');
INSERT INTO "rotation" VALUES(2, 'TRBM05-AFTER', 11.5, 3.14159265358979, 495, 'rotation');
INSERT INTO "rotation" VALUES(3, 'TRBM06-BEFORE', 12.5, 3.14159265358979, 498, 'rotation');
INSERT INTO "rotation" VALUES(4, 'TRBM06-AFTER', 13.5, 3.14159265358979, 500, 'rotation');
INSERT INTO "rotation" VALUES(5, 'TRBM07-BEFORE', 16.5, 3.14159265358979, 505, 'rotation');
INSERT INTO "rotation" VALUES(6, 'TRBM07-AFTER', 17.5, 3.14159265358979, 507, 'rotation');
INSERT INTO "rotation" VALUES(7, 'TRBM08-BEFORE', 18.5, 3.14159265358979, 510, 'rotation');
INSERT INTO "rotation" VALUES(8, 'TRBM08-AFTER', 19.5, 3.14159265358979, 512, 'rotation');
CREATE TABLE spch_comp(
  id integer primary key,
  name text,
  view_index double precision unique,
  fraction_model double precision not null default 1.0,
  model_index integer unique, 
  model_type text not null default 'spch_comp'
);
CREATE TABLE caperture(
  id integer primary key,
  name text,
  view_index double precision unique,
  aperture_model double precision not null default 0.0,
  in_out_model integer not null default 1,
  model_index integer unique, 
  model_type text not null default 'caperture'
);
CREATE TABLE raperture(
  id integer primary key,
  name text,
  view_index double precision unique,
  aperture_xl_model double precision not null default 0.0,
  aperture_xr_model double precision not null default 0.0,
  aperture_yt_model double precision not null default 0.0,
  aperture_yb_model double precision not null default 0.0,
  aperture_center_tmp double precision not null default 0.0,
  in_out_model integer not null default 1,
  channel integer references epics_channel(id)
          on delete restrict,
  channel2 integer references epics_channel(id)
          on delete restrict,
  model_index integer unique, 
  model_type text not null default 'raperture'
);
INSERT INTO "raperture" VALUES(1, 'TRBM05-APER', 11.75, 0.0696488118594079, 0.0343611881405921, 0.0261493, 0.0261493, 0.140891188140592, 1, 1, 2, 496, 'raperture');
INSERT INTO "raperture" VALUES(2, 'TRBM06-APER', 13.75, 0.113365177348903, 0.0484748226510966, 0.0261493, 0.0261493, 0.0508348226510966, 1, 1, 2, 501, 'raperture');
INSERT INTO "raperture" VALUES(3, 'TRBM07-APER', 17.75, 0.0696488118594079, 0.0343611881405921, 0.0261493, 0.0261493, 0.140891188140592, 1, 1, 2, 508, 'raperture');
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
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), 
    (select lcs_name from epics_channel where id = channel2), NULL, NULL from drift where channel != '' union
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), 
    (select lcs_name from epics_channel where id = channel2), NULL, NULL from raperture where channel != '' union
  select view_index, name, model_type, (select lcs_name from epics_channel where id = channel), 
    (select lcs_name from epics_channel where id = channel2), NULL, NULL from dipole;
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
              (select a14_cal from quad_family mf where quad.family_cal = mf.id)*
              power(((select case when abs(value) > thresh then abs(value) else 0.0 end from epics_channel where quad.channel=id)-shunt_cal), 14.0))/
              (select l_eff_cal from quad_family mf where quad.family_cal = mf.id))
    where exists(select * from epics_channel where quad.channel=new.rowid);
--- update dipole energy/momentum ----
  update dipole set
    bfield_ratio_tmp = (select value from epics_channel where dipole.channel = id)/ bfield_ratio_cal
    where exists(select * from epics_channel where dipole.channel = new.rowid);
--- update dipole rho ---
  update dipole set
    current_ratio_tmp = (select value from epics_channel where dipole.channel2 = id) / (select value from epics_channel where dipole.channel = id)
    where exists(select * from epics_channel where dipole.channel = new.rowid or dipole.channel2 = new.rowid);
end;
CREATE TRIGGER update_bfield_ratio AFTER UPDATE OF bfield_ratio_tmp ON dipole
begin
  update dipole set kenergy_model = mc2_design * (sqrt(1.0 + bfield_ratio_tmp *bfield_ratio_tmp* (kenergy_design/mc2_design) * (kenergy_design/mc2_design + 2.0)) - 1.0) 
    where rowid = new.rowid;
end;
CREATE TRIGGER update_current_ratio after update of current_ratio_tmp on dipole
begin
  update dipole set rho_model = (select case when current_ratio_tmp != 1 then 2.0 * 0.452 / sin(20.0/180.0*pi()) / (1.0 - current_ratio_tmp) else 1e20 end) where id == 2 or id == 5;
  update dipole set rho_model = 2.0 * 0.452 / sin(20.0/180.0*pi()) / (1.0 + current_ratio_tmp) where id == 3 or id == 4;
end;
CREATE TRIGGER update_rho after update of rho_model on dipole
begin
  update dipole set angle_model = 20.0/180.0*pi() - asin(sin(20.0/180.0*pi()) - 0.452/rho_model) where id == 2 or id == 5;
  update dipole set angle_model = asin(0.452/rho_model) where id == 3 or id == 4;
end;
CREATE TRIGGER update_angle after update of angle_model on dipole
begin
  update dipole set edge_angle2_model = -(20.0/180.0*pi() - angle_model) where id == 2;
  update dipole set edge_angle1_model = -(20.0/180.0*pi() - angle_model) where id == 5;
  update dipole set edge_angle1_model = angle_model where id == 3;
  update dipole set edge_angle2_model = angle_model where id == 4;
  update drift set length_model = dz_design/cos((select angle_model from dipole where id == 3)) where channel != '';
  update raperture set aperture_center_tmp = 0.01 * (21.885 - (45.2 * tan(20.0/180.0*pi() - 
        0.5*(select angle_model from dipole where id == 2)) - 0.5 * 45.2 * tan(20.0/180.0*pi())))
    where id == 1 or id == 3;
  update raperture set aperture_center_tmp = 0.01 * (21.885 - (45.2 * tan(0.5 * (select angle_model from dipole where id == 3)) +
        4.138 * tan((select angle_model from dipole where id == 3)) + 
        45.2 * tan(20.0/180.0*pi() - 0.5*(select angle_model from dipole where id == 2)) -
        0.5*45.2*tan(20.0/180.0*pi())))
    where id == 2;
end;
CREATE TRIGGER update_aperture_center after update of aperture_center_tmp on raperture
begin
  update raperture set aperture_xl_model = 0.2129 - aperture_center_tmp - 0.00236 where id == 1 or id == 3;
  update raperture set aperture_xr_model = aperture_center_tmp - 0.10417 - 0.00236 where id == 1 or id == 3;
  update raperture set aperture_xl_model = 0.16656 - aperture_center_tmp - 0.00236 where id == 2;
  update raperture set aperture_xr_model = aperture_center_tmp - 0.00236 where id == 2;
end;
COMMIT;
