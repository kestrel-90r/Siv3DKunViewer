
//
// Siv3D August 2016 v2 for Visual Studio 2019
// 
// Requirements
// - Visual Studio 2015 (v140) toolset
// - Windows 10 SDK (10.0.17763.0)
//

# include <Siv3D.hpp>

# define TINYGLTF_IMPLEMENTATION
# define STB_IMAGE_IMPLEMENTATION
# define TINYGLTF_NO_STB_IMAGE_WRITE
# include "3rd/tiny_gltf.h"

static tinygltf::Buffer* getBuffer(tinygltf::Model& model, tinygltf::Primitive& pr, int* offset)
{
    if (pr.indices == -1) return nullptr;
    auto& ai = model.accessors[pr.indices];
    auto& bi = model.bufferViews[ai.bufferView];
    auto& buf = model.buffers[bi.buffer];
    *offset = bi.byteOffset + ai.byteOffset;
    return &buf;
}

static tinygltf::Buffer* getBuffer(tinygltf::Model& model, tinygltf::Primitive& pr, const std::string attr, int* offset)
{
    if (pr.attributes.size() == 0) return nullptr;
    auto& ap = model.accessors[pr.attributes[attr]];
    auto& bp = model.bufferViews[ap.bufferView];
    auto& buf = model.buffers[bp.buffer];
    *offset = bp.byteOffset + ap.byteOffset;
    return &buf;
}

tinygltf::Model    model;       //glTFモデル
template <typename T> inline tinygltf::Value toVal(T value)
{
    return tinygltf::Value((std::vector<unsigned char>) reinterpret_cast<std::vector<unsigned char>&>(value));
}

inline tinygltf::Value toVal(Mat4x4* value, size_t size)
{
    return tinygltf::Value(reinterpret_cast<const unsigned char*>(value), size);
}

template <typename T> inline std::vector<double> toVec(T value)
{
    return reinterpret_cast<std::vector<double>&>(value.Get<std::vector<unsigned char>>());
}

inline Mat4x4 toMat(tinygltf::Value value)
{
    Mat4x4 mat = Mat4x4().Identity();
    if (value.IsBinary()) std::memcpy(&mat, reinterpret_cast<void*>(value.Get<std::vector<unsigned char>>().data()), sizeof(Mat4x4));
    return mat;
}

void gltfSetupPosture(tinygltf::Node& node);
void gltfSetupModel(tinygltf::Model& model)
{
    //シーンに含まれるモデル(ルートノード)を検索して子ノードを持っているものモデルルートとする。
    for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
    {
        auto& msn = model.nodes[model.scenes[0].nodes[nn]];

        //子ノード無ければスキップ※子ノードが無いノードは光源とかカメラ。
        if (0 == msn.children.size()) continue;

        //ノードにはメッシュノードとスケルトンノードの2種類が存在。
        for (int cc = 0; cc < msn.children.size(); cc++)
        {
            //glTF基本姿勢情報→姿勢変換行列を生成/登録
            gltfSetupPosture(model.nodes[msn.children[cc]]);
        }
    }
}

void gltfSetupPosture(tinygltf::Node& node)
{
    //extension領域にスケルトンアニメ用の姿勢行列を用意
    node.extensions["poserot"] = toVal(Array<double>{0, 0, 0, 1});
    node.extensions["posetra"] = toVal(Array<double>{0, 0, 0});
    node.extensions["posesca"] = toVal(Array<double>{1, 1, 1});
    node.extensions["weights"] = toVal(Array<double>{1, 1, 1});

    //glTFではすべてのノードがmodel.nodes[]に集約される。
    //ノードはスケルトンノードとメッシュノードの2種類がある。
    //全てのノードには１つの基本姿勢情報が含まれる。
    //メッシュノードにはMesh配列が含まれ再起しない。
    //スケルトンノードはツリー構造になっていて再起する。
    auto& rr = node.rotation;
    auto& tt = node.translation;
    auto& ss = node.scale;

    Quaternion rrr = rr.size() ? Quaternion(rr[0], rr[1], rr[2], rr[3]) : Quaternion(0, 0, 0, 1);
    Float3     ttt = tt.size() ? Float3(tt[0], tt[1], tt[2]) : Float3(0, 0, 0);
    Float3     sss = ss.size() ? Float3(ss[0], ss[1], ss[2]) : Float3(1, 1, 1);

    //glTF基本姿勢情報→姿勢変換行列を生成(SRT)
    Mat4x4 matlocal = Mat4x4().Identity().Scale(sss) *      //拡縮
                      rrr.toMatrix() *                      //回転
                      Mat4x4().Identity().Translate(ttt);   //移動

    //extensions領域(by tinyGLTF)をテンポラリとして使用して姿勢変換行列を保存
    node.extensions["matlocal"] = toVal(&matlocal, sizeof(Mat4x4));

    //子ノードを再起についても同様に姿勢変換行列を保存
    for (int cc = 0; cc < node.children.size(); cc++)
        gltfSetupPosture(model.nodes[node.children[cc]]);
}

void gltfCalcSkeleton(tinygltf::Node& node, Mat4x4& matparent)
{
    if (node.mesh >= 0) return; //メッシュノードは処理しない

    //extension領域のスケルトンアニメ用の姿勢行列を取得
    //※まだアニメ実装まだなので初期値固定
    Array<double> rr = toVec(node.extensions["poserot"]);
    Array<double> tt = toVec(node.extensions["posetra"]);
    Array<double> ss = toVec(node.extensions["posesca"]);

    Quaternion rrr = rr.size() ? Quaternion(rr[0], rr[1], rr[2], rr[3]) : Quaternion(0, 0, 0, 1);
    Float3     ttt = tt.size() ? Float3(tt[0], tt[1], tt[2]) : Float3(0, 0, 0);
    Float3     sss = ss.size() ? Float3(ss[0], ss[1], ss[2]) : Float3(1, 1, 1);

    //アニメ姿勢情報→アニメ姿勢変換行列(matpose)を生成(SRT)
    Mat4x4 posesca = Mat4x4().Identity().Scale(sss);    //拡縮
    Mat4x4 poserot = rrr.toMatrix();                    //回転
    Mat4x4 posetra = Mat4x4().Identity().Translate(ttt);//移動
    Mat4x4 matpose = posesca * poserot * posetra;

    //基本姿勢変換行列を取得
    Mat4x4 matlocal = toMat(node.extensions["matlocal"]);

    //アニメ姿勢変換行列(matpose)が単位行列の場合は基本姿勢変換行列のまま
    if (0 == std::memcmp(&Mat4x4().Identity(), &matpose, sizeof(Mat4x4)))
        matpose = matlocal;

    //基本姿勢変換行列を原点に移動してアニメ姿勢変換行列(matpose)で座標変換
    Mat4x4 mat = matpose * matlocal.inverse();

    //原点から基本姿勢変換行列を適用して親ノード行列合わせてワールド座標の姿勢変換行列を生成
    Mat4x4 matworld = mat * matlocal * matparent;

    //extensions領域(by tinyGLTF)にアニメ姿勢変換行列(world)を保存
    node.extensions["matworld"] = toVal(&matworld, sizeof(Mat4x4));

    //子ノードについても再起でアニメ姿勢変換行列(world)を保存
    for (int cc = 0; cc < node.children.size(); cc++)
        gltfCalcSkeleton(model.nodes[node.children[cc]], matworld);
}

Array<Mat4x4> matjoints;  //スケルトンのボーン行列を確保

void gltfUpdateSkin(tinygltf::Node& node)
{
    if (node.mesh >= 0 && node.skin >= 0)  //メッシュノードのみ処理
    {
        //このメッシュノードに該当するスケルトン情報(skin)から逆バインド行列を取得
        auto& msns = model.skins[node.skin];
        auto& ibma = model.accessors[msns.inverseBindMatrices];
        auto& ibmbv = model.bufferViews[ibma.bufferView];
        auto  ibmd = model.buffers[ibmbv.buffer].data.data() + ibma.byteOffset + ibmbv.byteOffset;

        //モデルのワールド座標の行列を作成
        Mat4x4 matmodel = Mat4x4().Identity();

        //スケルトンの全ボーン(glTF的にはjoint)を破棄
        matjoints.clear();

        //スケルトンのボーンにアニメ姿勢変換行列(world)を適用
        for (int ii = 0; ii < msns.joints.size(); ii++)
        {
            //ボーンのスケルトンノードを取得
            auto& skeletonode = model.nodes[msns.joints[ii]];

            //ボーンに該当する逆バインド行列を取得をBufferViewから取得
            Mat4x4 ibm = *(Mat4x4*)&ibmd[ii * sizeof(Mat4x4)];

            //アニメ姿勢変換行列(world)を取得
            Mat4x4 matworld = toMat(skeletonode.extensions["matworld"]);

            //モデルのワールド座標の行列を原点に戻してアニメ姿勢変換行列(world)を適用してIBM
            Mat4x4 matjoint = ibm * matworld * matmodel.inverse();

            //スケルトンのボーン情報に追加
            matjoints.emplace_back(matjoint);
        }
    }
}

void gltfDrawMesh(tinygltf::Node& node);
void gltfDrawModel(tinygltf::Model& model)
{
    //シーンに含まれるモデル(ルートノード)を検索して子ノードを持っているものモデルルートとする。
    //※子ノードが無いノードは光源とかカメラ。
    for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
    {
        auto& msn = model.nodes[model.scenes[0].nodes[nn]];

        //子ノード無ければスキップ※子ノードが無いノードは光源とかカメラ。
        if (0 == msn.children.size()) continue;

        //モデルのスケルトンノードだけ先に処理
        for (int cc = 0; cc < msn.children.size(); cc++)
        {
            auto& node = model.nodes[msn.children[cc]];

            //子ノードを再起で基本姿勢計算１
            gltfCalcSkeleton( node, Mat4x4().Identity());//姿勢制御用行列を初期化（スケルトンノード）
        }

        //モデルのメッシュノードを探して描画処理
        for (int cc = 0; cc < msn.children.size(); cc++)
        {
            auto& node = model.nodes[msn.children[cc]];

            //メッシュノードからスキンを更新（メッシュノードは再起なし）
            gltfUpdateSkin(node);                       //姿勢制御用行列を初期化（メッシュノード）

            if (node.mesh == -1)//スケルトンノードの場合は子ノード検索して描画
            {
                for (int ccc = 0; ccc < node.children.size(); ccc++)
                    gltfDrawMesh(model.nodes[node.children[ccc]]);
            }
            else               //メッシュノードの場合はそのまま描画　
            {
                gltfDrawMesh(node);
            }
        }
    }
}

using Word4 = Vector4D<unsigned short>;
void gltfDrawMesh(tinygltf::Node& node)
{
    if (node.mesh == -1)//スケルトンノードの場合は子ノード検索して描画
    {
        for (int ccc = 0; ccc < node.children.size(); ccc++)
            gltfDrawMesh(model.nodes[node.children[ccc]]);
    }
    else               //メッシュノードの場合　
    {
        //メッシュノードに含まれる全プリミティブを描画
        for (int pp = 0; pp < model.meshes[node.mesh].primitives.size(); pp++)
        {
            auto& pr = model.meshes[node.mesh].primitives[pp];

            //glTFのBufferから頂点情報を取得
            int opos, otex, onormal, ojoints, oweights, oidx;
            auto& bpos = *getBuffer(model, pr, "POSITION", &opos);
            auto& btex = *getBuffer(model, pr, "TEXCOORD_0", &otex);
            auto& bnormal = *getBuffer(model, pr, "NORMAL", &onormal);
            auto& bjoint = *getBuffer(model, pr, "JOINTS_0", &ojoints);
            auto& bweight = *getBuffer(model, pr, "WEIGHTS_0", &oweights);
            auto& bidx = *getBuffer(model, pr, &oidx);  //頂点インデックス

            //頂点情報からMeshVertex配列を生成
            Array<MeshVertex> vertices;
            auto& map = model.accessors[pr.attributes["POSITION"]];
            for (int vv = 0; vv < map.count; vv++)
            {
                float* vertex = nullptr, * texcoord = nullptr, * normal = nullptr;
                vertex = (float*)&bpos.data.at(vv * 12 + opos);         //頂点座標
                texcoord = (float*)&btex.data.at(vv * 8 + otex);        //テクスチャ座標
                normal = (float*)&bnormal.data.at(vv * 12 + onormal);   //法線

                //基本姿勢のMeshVertex配列を生成
                MeshVertex mv;
                mv.position = Float3(vertex[0], vertex[1], vertex[2]);
                mv.texcoord = Float2(texcoord[0], texcoord[1]);
                mv.normal = Float3(normal[0], normal[1], normal[2]);

                //スキニング情報がある
                if (pr.attributes["JOINTS_0"] && pr.attributes["WEIGHTS_0"])
                {
                    //スケルトンのボーン情報がある
                    Mat4x4 matskin = Mat4x4().Identity();
                    if (matjoints.size())
                    {
                        auto joint = (uint16_t*)&bjoint.data.at(vv * 8 + ojoints);  //1頂点あたり4ボーンのIDと
                        auto weight = (float*)&bweight.data.at(vv * 16 + oweights); //4ボーンのウェイト取得
                        Word4  j4 = Word4(joint[0], joint[1], joint[2], joint[3]);
                        Float4 w4 = Float4(weight[0], weight[1], weight[2], weight[3]);

                        //4つのボーンにウェイト掛けて合成し頂点スキニング行列を生成
                        matskin = w4.x * matjoints[j4.x] +
                                  w4.y * matjoints[j4.y] +
                                  w4.z * matjoints[j4.z] +
                                  w4.w * matjoints[j4.w];
                    }

                    //頂点座標と法線ベクトルに頂点スキニング行列を適用
                    auto matpos = matskin.transform(Float4(mv.position, 1.0f));
                    auto matnor = matskin.inverse().transposed();

                    //頂点スキニング済のMeshVertex配列を生成
                    MeshVertex smv;
                    smv.position = Float3(matpos.x, matpos.y, matpos.z) / matpos.w;
                    smv.normal = matnor.transform(mv.normal);

                    //テクスチャのUV座標を生成
                    if (pr.attributes["TEXCOORD_0"]) smv.texcoord = Float2(texcoord[0], texcoord[1]);

                    //頂点情報として格納
                    vertices.emplace_back(smv);
                }
                else //スキニングなしの場合は基本姿勢の頂点情報を格納
                {
                    vertices.emplace_back(mv);
                }
            }

            MeshData md;    //Siv3Dメッシュデータ

            //頂点インデックス情報がある
            if (pr.indices > 0)
            {
                Array<uint32_t> indices;                    //インデックス配列
                auto& mapi = model.accessors[pr.indices];   //glTFから頂点インデックスを取得
                for (int ii = 0; ii < mapi.count; ii++)
                {
                    uint32_t idx;   //16bitと32bitの２つの頂点インデックスタイプがある
                    idx = (mapi.componentType == 5123) ? *(uint16_t*)&bidx.data.at(ii * 2 + oidx) :   //16bitINDISE
                          (mapi.componentType == 5125) ? *(uint32_t*)&bidx.data.at(ii * 4 + oidx) : 0;//32bitINDISE

                    indices.emplace_back(idx);              //インデックス配列に格納
                }

                md = MeshData(vertices, indices);           //頂点情報と頂点インデックスから描画用メッシュデータを生成

                vertices.clear();
                indices.clear();
            }

            auto rot = Quaternion().RollPitchYaw(0, 0, -3.14 / 2);
            auto tra = Float3(0, 0, 0);
            auto sca = Float3(5, 5, 5);

            //テクスチャを登録（※描画ごとに同じテクスチャ登録は無駄）
            //Meshes->Primitive->Metarial->baseColorTexture.json_double_value.index->images
            if (pr.material >= 0)
            {
                auto& nt = model.materials[pr.material].additionalValues["normalTexture"];  //法線マップを取得
                int idx = -1;
                auto& mmv = model.materials[pr.material].values;    //マテリアル情報を取得
                auto& bcf = mmv["baseColorFactor"];                 //頂点色情報を取得
                if (mmv.count("baseColorTexture"))                  //テクスチャIDを取得
                    idx = model.textures[(int)mmv["baseColorTexture"].json_double_value["index"]].source;

                //頂点色を登録（※描画毎に同じ頂点色登録は無駄）
                ColorF col(1, 1, 1, 1);                             //とりあえず白、頂点色有れば更新
                if (bcf.number_array.size())
                    col = ColorF(bcf.number_array[0], bcf.number_array[1], bcf.number_array[2], bcf.number_array[3]);

                //materials->textures->images
                if (idx >= 0 && model.images.size())                //テクスチャがある
                {
                    Texture tex;
                    if (model.images[idx].bufferView >= 0)          //glbファイルの場合はBufferからテクスチャを取得
                    {
                        auto& bgfx = model.bufferViews[model.images[idx].bufferView];
                        auto bimg = &model.buffers[bgfx.buffer].data.at(bgfx.byteOffset);

                        ByteArray teximg((void*)bimg, bgfx.byteLength);
                        tex = Texture(std::move(teximg), TextureDesc::For3D);
                    }
                    else                                            //glTFファイルの場合はiamges直下からテクスチャを取得
                    {
                        auto& mii = model.images[idx].image;
                        ByteArray teximg((void*)&mii, mii.size());
                        tex = Texture(std::move(teximg), TextureDesc::For3D);
                    }
                    DynamicMesh(md).rotated(rot).scaled(sca).translated(tra).draw(tex, col);//テクスチャ付きメッシュを描画
                }
                else DynamicMesh(md).rotated(rot).scaled(sca).translated(tra).draw(col);    //頂点色メッシュを描画
            }
            else DynamicMesh(md).rotated(rot).scaled(sca).translated(tra).draw();           //マテリアル無いメッシュを描画
        }
    }
}

using namespace s3d::Input;

void Main()
{
    Profiler::EnableWarning(false); //アセット大量破棄警告を無効

    const Font fontS(10);            //フォントサイズ指定
    Window::Resize(800, 450);       //画面サイズ
    auto ww = Window::Width();
    auto hh = Window::Height();
    Graphics::SetBackground(Palette::Dimgray);

    tinygltf::TinyGLTF loader;      //Siv3Dkunモデルを読み込み
    std::string err, warn;
    loader.LoadBinaryFromFile(&model, &err, &warn, "Siv3Dkun.glb");

    gltfSetupModel(model);          //glTFモデル準備

    Stopwatch stopwatch;
    stopwatch.start();               //ストップウォッチ開始

    while (System::Update())
    {
        Graphics3D::FreeCamera();

        auto min = stopwatch.min();
        auto sec = stopwatch.s() - min * 60;
        auto ms = stopwatch.ms() - stopwatch.s() * 1000;

        String t1 = L"#1分:#2秒:#3ms経過";
        fontS(t1.replace(L"#1", Pad(min, { 3,L'0' }))
                .replace(L"#2", Pad(sec, { 2,L'0' }))
                .replace(L"#3", Pad(ms,  { 3,L'0' }))).draw(ww - 200, hh - 55);

        auto qt = Quaternion().rollPitchYaw(0, 0, 0);
        auto ms1 = stopwatch.ms();

        gltfDrawModel(model);       //glTFモデル描画

        auto ms2 = stopwatch.ms();

        String t2 = L"Render:(1)ms"; 
        fontS( t2.replace(L"(1)", Pad(ms2 - ms1, { 5,L' ' })) ).draw(ww - 200, hh - 35);
    }
}
