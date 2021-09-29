organ="spleen"
# liver
# spleen
# pancreas

endpoint="https://s3.twcc.ai:443"
access_key="5QL09M2O1Y8E4GTOFC9Z"
secret_key="9mXMT1kJAYAzOGZusIc5CT856cc3O22FqaYZpeTN"
bucket="test-bucket-clara"
output_s3_folder="Inference_Source/2021-09-02_13:31:07.303"

curl -X POST http://localhost:1234/clara/$organ --header "Content-Type: application/json" --data "{\"endpoint\" : \"$endpoint\", \"access_key\" : \"$access_key\", \"secret_key\" : \"$secret_key\", \"bucket\" : \"$bucket\", \"output_s3_folder\": \"$output_s3_folder\", \"asset_group\": [{\"category_name\": \"${organ}_seg_1\", \"files\": [\"advantech_aifs_aiaa/01160825/960930_68431567.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431581.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431595.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431609.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431623.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431637.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431568.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431582.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431596.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431610.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431624.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431638.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431569.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431583.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431597.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431611.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431625.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431639.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431570.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431584.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431598.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431612.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431626.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431640.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431571.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431585.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431599.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431613.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431627.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431641.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431572.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431586.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431600.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431614.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431628.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431642.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431573.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431587.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431601.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431615.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431629.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431643.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431574.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431588.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431602.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431616.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431630.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431644.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431575.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431589.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431603.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431617.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431631.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431645.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431576.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431590.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431604.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431618.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431632.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431646.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431577.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431591.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431605.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431619.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431633.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431647.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431578.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431592.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431606.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431620.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431634.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431579.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431593.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431607.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431621.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431635.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431580.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431594.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431608.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431622.dcm\", \"advantech_aifs_aiaa/01160825/960930_68431636.dcm\"]}, {\"category_name\": \"${organ}_seg_2\", \"files\": [\"advantech_aifs_aiaa/01160825_2/960930_68431567.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431581.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431595.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431609.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431623.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431637.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431568.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431582.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431596.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431610.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431624.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431638.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431569.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431583.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431597.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431611.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431625.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431639.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431570.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431584.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431598.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431612.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431626.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431640.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431571.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431585.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431599.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431613.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431627.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431641.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431572.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431586.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431600.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431614.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431628.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431642.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431573.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431587.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431601.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431615.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431629.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431643.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431574.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431588.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431602.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431616.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431630.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431644.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431575.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431589.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431603.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431617.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431631.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431645.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431576.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431590.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431604.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431618.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431632.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431646.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431577.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431591.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431605.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431619.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431633.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431647.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431578.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431592.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431606.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431620.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431634.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431579.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431593.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431607.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431621.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431635.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431580.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431594.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431608.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431622.dcm\", \"advantech_aifs_aiaa/01160825_2/960930_68431636.dcm\"]}, {\"category_name\": \"${organ}_seg_3\", \"files\": [\"advantech_aifs_aiaa/01160825_3/960930_68431567.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431581.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431595.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431609.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431623.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431637.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431568.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431582.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431596.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431610.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431624.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431638.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431569.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431583.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431597.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431611.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431625.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431639.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431570.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431584.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431598.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431612.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431626.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431640.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431571.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431585.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431599.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431613.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431627.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431641.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431572.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431586.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431600.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431614.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431628.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431642.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431573.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431587.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431601.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431615.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431629.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431643.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431574.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431588.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431602.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431616.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431630.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431644.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431575.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431589.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431603.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431617.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431631.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431645.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431576.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431590.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431604.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431618.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431632.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431646.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431577.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431591.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431605.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431619.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431633.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431647.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431578.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431592.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431606.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431620.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431634.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431579.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431593.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431607.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431621.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431635.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431580.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431594.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431608.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431622.dcm\", \"advantech_aifs_aiaa/01160825_3/960930_68431636.dcm\"]}]}"
