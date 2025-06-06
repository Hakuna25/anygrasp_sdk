# License Registration

- Get the feature id of your working machine.
```base
    ./license_checker -f
```
- Fill in the [form](https://forms.gle/XVV3Eip8njTYJEBo6) to apply for license, which requires the machine feature id. **Note: if there is a "%" at the end of the machine id, delete it when filling the form.**
- You will get a `.zip` file that contains license. The folder structure is as follows (see [sample_license](sample_license) for example):
```base
    license/
       |-- licenseCfg.json
       |-- [your_name].public_key
       |-- [your_name].signature
       |-- [your_name].lic
```
- You can check license states via
```base
    ./license_checker -c license/licenseCfg.json
```
- Put the license folder under [grasp_detection](../grasp_detection) and [grasp_tracking](../grasp_tracking) to execute the demo.
