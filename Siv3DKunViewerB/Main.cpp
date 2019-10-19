
//
// Siv3D August 2016 v2 for Visual Studio 2019
// 
// Requirements
// - Visual Studio 2015 (v140) toolset
// - Windows 10 SDK (10.0.17763.0)
//

# include <Siv3D.hpp>

//#define STOPMOTION       //Todo:スプラインバグ調査用コマ送り確認

using namespace s3d::Input;

# define TINYGLTF_IMPLEMENTATION
# define STB_IMAGE_IMPLEMENTATION
# define TINYGLTF_NO_STB_IMAGE_WRITE
# include "3rd/tiny_gltf.h"

using Word4 = Vector4D<unsigned short>;

struct Channel
{
    int8_t  typeDelta;      // 1:Translation 2:Scale 3:Rotation 4:Weight
    Array<std::pair<int, Float4>> deltaKeyframes;
    int32_t idxSampler;
    int32_t idxNode;
    Channel() { typeDelta = 0; idxSampler = -1; idxNode = -1; }
};

struct Sampler
{
    Array<std::pair<int, float>> interpolateKeyframes;
    float  minTime;
    float  maxTime;
    Sampler() { minTime = 0.0; maxTime = 0.0; }
};

using Word4 = Vector4D<uint16_t>;

struct Frame
{
    Array<DynamicMesh>    Meshes;     // [primitive]メッシュデータ
    Array<int>            TexCol;     // [primitive]テクスチャ頂点色識別子
    Array<Sampler>        Samplers;   // [primitive]補間形式(step,linear,spline)、前後フレーム、データ形式
    Array<Channel>        Channels;   // [primitive]デルタ(移動,回転,拡縮,ウェイト)
};

struct PrecomputeAnime      //model.animationsに対応
{
    size_t                currentframe;
    Array<ColorF>         meshColors; // [primitive]頂点色データ
    Array<Texture>        meshTexs;   // [primitive]テクスチャデータ
                                      // ※頂点色とテクスチャは全フレームで共通
    Array<Frame>          Frames;     // [フレーム]
    Array< Array<Mat4x4>> Joints;     // [Skin][joint]
};

Array <PrecomputeAnime> precAnimes; //glTFコンテナ

tinygltf::Model    model;           //glTFモデルローダー

void gltfInterpolateStep(Channel& ch, int lowframe);
void gltfInterpolateLinear(Channel& ch, int lowframe, int uppframe, float weight);
void gltfInterpolateSpline(Channel& ch, int lowframe, int uppframe, float lowtime, float upptime, float weight);
void gltfCalcSkeleton(tinygltf::Node& node, Mat4x4& matparent);
void gltfPrecomputeMesh(int idxframe, PrecomputeAnime& anime, tinygltf::Node& node);

tinygltf::Buffer* getBuffer(tinygltf::Model& model, tinygltf::Primitive& pr, int* offset)
{
    if (pr.indices == -1) return nullptr;
    auto& ai = model.accessors[pr.indices];
    auto& bi = model.bufferViews[ai.bufferView];
    auto& buf = model.buffers[bi.buffer];
    *offset = bi.byteOffset + ai.byteOffset;
    return &buf;
}

tinygltf::Buffer* getBuffer(tinygltf::Model& model, tinygltf::Primitive& pr, const std::string attr, int* offset)
{
    if (pr.attributes.size() == 0) return nullptr;
    auto& ap = model.accessors[pr.attributes[attr]];
    auto& bp = model.bufferViews[ap.bufferView];
    auto& buf = model.buffers[bp.buffer];
    *offset = bp.byteOffset + ap.byteOffset;
    return &buf;
}

template <typename T> tinygltf::Value toVal(T value)
{
    return tinygltf::Value((std::vector<unsigned char>) reinterpret_cast<std::vector<unsigned char>&>(value));
}

tinygltf::Value toVal(Float4* value, size_t size)
{
    return tinygltf::Value(reinterpret_cast<const unsigned char*>(value), size);
}

tinygltf::Value toVal(Mat4x4* value, size_t size)
{
    return tinygltf::Value(reinterpret_cast<const unsigned char*>(value), size);
}

template <typename T> std::vector<float> toFloat(T value)
{
    return reinterpret_cast<std::vector<float>&>(value.Get<std::vector<unsigned char>>());
}

Mat4x4 toMat(tinygltf::Value value)
{
    Mat4x4 mat = Mat4x4().Identity();
    if (value.IsBinary())
        std::memcpy(&mat, reinterpret_cast<void*>(value.Get<std::vector<unsigned char>>().data()), sizeof(Mat4x4));
    return mat;
}

void gltfSetupPosture(tinygltf::Node& node)
{
    //extensionにスケルトンアニメ用の姿勢行列を用意して制御に利用
    node.extensions["poserot"] = toVal(&Float4(0, 0, 0, 1), sizeof(Float4));
    node.extensions["posetra"] = toVal(&Float4(0, 0, 0, 0), sizeof(Float4));
    node.extensions["posesca"] = toVal(&Float4(1, 1, 1, 0), sizeof(Float4));
    node.extensions["posewei"] = toVal(&Float4(0, 0, 0, 0), sizeof(Float4));

    //    node.extensions["test"] = toVal(&Float4(1, 2, 3, 4), sizeof(Float4));
    //    auto test = toFloat4(node.extensions["test"]);

    auto& r = node.rotation;
    auto& t = node.translation;
    auto& s = node.scale;

    Quaternion rr = r.size() ? Quaternion(r[0], r[1], r[2], r[3]) : Quaternion(0, 0, 0, 1);
    Float3     tt = t.size() ? Float3(t[0], t[1], t[2]) : Float3(0, 0, 0);
    Float3     ss = s.size() ? Float3(s[0], s[1], s[2]) : Float3(1, 1, 1);

    //OpenGL系からSiv3Dへの行列計算はSRTの順
    Mat4x4 matlocal = Mat4x4().Identity().Scale(ss) *
        rr.toMatrix() *
        Mat4x4().Identity().Translate(tt);

    //基本姿勢登録
    node.extensions["matlocal"] = toVal(&matlocal, sizeof(Mat4x4));

    //子ノードを再起で基本姿勢登録
    for (int cc = 0; cc < node.children.size(); cc++)
        gltfSetupPosture(model.nodes[node.children[cc]]);
}

int dbgAA;

void gltfSetupModel(tinygltf::Model& model,int cycleframe )
{
    //シーンに含まれるモデル(ルートノード)を検索して子ノードを持っているものモデルルートとする。
    for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
    {
        auto& msn = model.nodes[model.scenes[0].nodes[nn]];
        if (msn.children.size()) //モデルノード
        {
            //モデルノードにはメッシュルートノードとスケルトンルートノードの2種類
            for (int cc = 0; cc < msn.children.size(); cc++)
            {
                //メッシュルートノードにはmeshが有る。スケルトンルートノードには無い。
                //スケルトンルートノードは同じnodesを通じてノード階層を参照。

                //子ノードを再起で基本姿勢登録
                gltfSetupPosture(model.nodes[msn.children[cc]]);
            }
        }
    }

    const auto aid = 0;                      // 現状アニメは１つのみ0固定
    auto& man = model.animations[aid];       // ※ここで例外出たらglbファイル開くが失敗している。

    auto cycletime = 1.0f;                   // 1秒を想定
    auto frametime = cycletime / cycleframe; // 1フレーム時間
    auto currenttime = 0.0f;                 // 現在時刻

    auto& macc = model.accessors;
    auto& mbv = model.bufferViews;
    auto& mb = model.buffers;

    precAnimes.resize(model.animations.size());

    //全フレームの補間形式(step,linear,3dspline)、フレーム時刻、delta(移動,回転,拡縮,ウェイト)を収集
    precAnimes[aid].Frames.resize(cycleframe);

    for (auto aa = 0; aa < cycleframe; aa++)
    {
        dbgAA = aa;

        auto& mas = model.animations[aid].samplers;
        auto& mac = model.animations[aid].channels;
        auto& as = precAnimes[aid].Frames[aa].Samplers;
        auto& ac = precAnimes[aid].Frames[aa].Channels;
        as.resize(mas.size());
        ac.resize(mac.size());

        // GLTFサンプラを取得
        for (auto ss = 0; ss < mas.size(); ss++)
        {
            auto& masi = model.accessors[mas[ss].input];  // 前フレーム情報

            as[ss].minTime = 0;
            as[ss].maxTime = 1;
            if (masi.minValues.size() > 0 && masi.maxValues.size() > 0)
            {
                as[ss].minTime = float(masi.minValues[0]);
                as[ss].maxTime = float(masi.maxValues[0]);
            }

            as[ss].interpolateKeyframes.resize(masi.count);

            for (auto kk = 0; kk < masi.count; kk++)
            {
                auto& sai = mas[ss].input;
                const auto& offset = mbv[macc[sai].bufferView].byteOffset + macc[sai].byteOffset;
                const auto& stride = masi.ByteStride(mbv[masi.bufferView]);
                void* adr = &mb[mbv[macc[sai].bufferView].buffer].data.at(offset + kk * stride);

                auto& ctype = macc[sai].componentType;
                float value = (ctype == 5126) ? *(float*)adr :
                              (ctype == 5123) ? *(uint16_t*)adr :
                              (ctype == 5121) ? *(uint8_t*)adr :
                              (ctype == 5122) ? *(int16_t*)adr :
                              (ctype == 5120) ? *(int8_t*)adr : 0.0;

                as[ss].interpolateKeyframes[kk] = std::make_pair(kk, value);
            }
        }

        // GLTFチャネルを取得
        for (int cc = 0; cc < ac.size(); cc++)
        {
            auto& maso = model.accessors[mas[cc].output];
            auto& macso = macc[mas[mac[cc].sampler].output];
            const auto& stride = maso.ByteStride(mbv[maso.bufferView]);
            const auto& offset = mbv[macso.bufferView].byteOffset + macso.byteOffset;

            ac[cc].deltaKeyframes.resize(maso.count);
            ac[cc].idxNode = mac[cc].target_node;
            ac[cc].idxSampler = mac[cc].sampler;

            if (mac[cc].target_path == "weights")
            {
                ac[cc].typeDelta = 4;   //weight

                for (int ff = 0; ff < maso.count; ff++)
                {
                    void* adr = (void*)&mb[mbv[macso.bufferView].buffer].data.at(offset + ff * stride);

                    auto& ctype = macso.componentType;
                    float value = (ctype == 5126) ? *(float*)adr :
                                  (ctype == 5123) ? *(uint16_t*)adr :
                                  (ctype == 5121) ? *(uint8_t*)adr :
                                  (ctype == 5122) ? *(int16_t*)adr :
                                  (ctype == 5120) ? *(int8_t*)adr : 0.0;

                    ac[cc].deltaKeyframes[ff].first = ff;
                    ac[cc].deltaKeyframes[ff].second = Float4(value, 0, 0, 0);
                }
            }

            if (mac[cc].target_path == "translation")
            {
                ac[cc].typeDelta = 1;   //translate

                for (int ff = 0; ff < maso.count; ff++)
                {
                    float* tra = (float*)&mb[mbv[macso.bufferView].buffer].data.at(offset + ff * stride);
                    ac[cc].deltaKeyframes[ff].first = ff;
                    ac[cc].deltaKeyframes[ff].second = Float4(tra[0], tra[1], tra[2], 0);
                }
            }

            if (mac[cc].target_path == "rotation")
            {
                ac[cc].typeDelta = 3;
                for (int ff = 0; ff < maso.count; ff++)
                {
                    float* rot = (float*)&mb[mbv[macso.bufferView].buffer].data.at(offset + ff * stride);
                    auto qt = Quaternion(rot[0], rot[1], rot[2], rot[3]).normalize();

                    ac[cc].deltaKeyframes[ff].first = ff;
                    ac[cc].deltaKeyframes[ff].second = Float4(qt.component.m128_f32[0],
                                                              qt.component.m128_f32[1],
                                                              qt.component.m128_f32[2],
                                                              qt.component.m128_f32[3]);
                }
            }

            if (mac[cc].target_path == "scale")
            {
                ac[cc].typeDelta = 2;
                for (int ff = 0; ff < maso.count; ff++)
                {
                    float* sca = (float*)&mb[mbv[macso.bufferView].buffer].data.at(offset + ff * stride);
                    ac[cc].deltaKeyframes[ff].first = ff;
                    ac[cc].deltaKeyframes[ff].second = Float4(sca[0], sca[1], sca[2], 0);
                }
            }
        }

        // 姿勢確定
        for (auto& ch : precAnimes[aid].Frames[aa].Channels)
        {
            auto& sa = precAnimes[aid].Frames[aa].Samplers[ch.idxSampler];

            std::pair<int, float> f0, f1;
            size_t kf;
            for (kf = 0; kf < sa.interpolateKeyframes.size() - 1; kf++)
            {
                f0 = sa.interpolateKeyframes[kf];
                f1 = sa.interpolateKeyframes[kf + 1];
                if (f0.second <= currenttime && f1.second >= currenttime) break;
            }
            if (kf == sa.interpolateKeyframes.size()) break;


            float lowtime = f0.second;
            float upptime = f1.second;
            const int lowframe = f0.first;
            const int uppframe = f1.first;
            auto& interpol = mas[ch.idxSampler].interpolation;

            // 再生時刻を正規化してキーフレーム間ウェイト算出
            const float mix = (currenttime - lowtime) / (upptime - lowtime);

            //キーフレーム間ウェイト、補間モード、下位フレーム/時刻、上位フレーム/時刻から姿勢確定
            if      (interpol == "STEP")        gltfInterpolateStep(ch, lowframe);
            else if (interpol == "LINEAR")      gltfInterpolateLinear(ch, lowframe, uppframe, mix);
            else if (interpol == "CUBICSPLINE") gltfInterpolateSpline(ch, lowframe, uppframe, lowtime, upptime, mix);

//          _RPTN(0, "[%04d]:time=%.04f(%.04f) \n", aa, currenttime, weight);
        }

        for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
        {
            auto& msn = model.nodes[model.scenes[0].nodes[nn]];
            for (int cc = 0; cc < msn.children.size(); cc++)
            {
                auto& node = model.nodes[msn.children[cc]];
                gltfCalcSkeleton(node, Mat4x4().Identity());
            }
        }

        Mat4x4 matinverse = Mat4x4().Identity();        //この記述は適当

        precAnimes[aid].Joints.clear();                 //1モデル専用になってしまっている
        precAnimes[aid].Joints.resize(model.skins.size());

        for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
        {
            auto& msn = model.nodes[model.scenes[0].nodes[nn]];
            for (int cc = 0; cc < msn.children.size(); cc++)
            {
                auto& node = model.nodes[msn.children[cc]];
                if (node.skin < 0) continue;         //LightとCameraをスキップ

                auto& msns = model.skins[node.skin];
                auto& ibma = model.accessors[msns.inverseBindMatrices];
                auto& ibmbv = model.bufferViews[ibma.bufferView];
                auto  ibmd = model.buffers[ibmbv.buffer].data.data() + ibma.byteOffset + ibmbv.byteOffset;

                //Meshを参照するノードはskinを参照する
                for (int ii = 0; ii < msns.joints.size(); ii++)
                {
                    //ボーンのスケルトンノードを取得
                    auto& skeletonode = model.nodes[msns.joints[ii]];
                    Mat4x4 ibm = *(Mat4x4*)&ibmd[ii * sizeof(Mat4x4)];
                    Mat4x4 matworld = toMat(skeletonode.extensions["matworld"]);
                    Mat4x4 matjoint = ibm * matworld * matinverse;

                    precAnimes[aid].Joints[node.skin].emplace_back(matjoint);
                }
            }
        }

        for (int nn = 0; nn < model.scenes[0].nodes.size(); nn++)
        {
            auto& msn = model.nodes[model.scenes[0].nodes[nn]];
            for (int cc = 0; cc < msn.children.size(); cc++)
            {
                auto& node = model.nodes[msn.children[cc]];
                if (node.mesh >= 0)//メッシュノード　
                {
                    //現在のノード(メッシュ)を描画(プリコンピュート)
                    gltfPrecomputeMesh(aa, precAnimes[aid], node);
                }
            }
        }

        currenttime += frametime;   //次のフレーム時刻に進める
    }
}

void gltfInterpolateStep(Channel& ch, int lowframe)
{
    Float4 val = ch.deltaKeyframes[lowframe].second;
    if (ch.typeDelta == 1)      model.nodes[ch.idxNode].extensions["posetra"] = toVal(&val);
    else if (ch.typeDelta == 3) model.nodes[ch.idxNode].extensions["poserot"] = toVal(&val);
    else if (ch.typeDelta == 2) model.nodes[ch.idxNode].extensions["posesca"] = toVal(&val);
    else if (ch.typeDelta == 4) model.nodes[ch.idxNode].extensions["posewei"] = toVal(&val);
}

void gltfInterpolateLinear(Channel& ch, int lowframe, int uppframe, float tt)
{
    Float4 low = ch.deltaKeyframes[lowframe].second;
    Float4 upp = ch.deltaKeyframes[uppframe].second;
    if (ch.typeDelta == 4)
    {
        //TBD weight
    }
    else if (ch.typeDelta == 1)//Translation
    {
        Float4 val = low * (1.0 - tt) + upp * tt;
        model.nodes[ch.idxNode].extensions["posetra"] = toVal(&val, sizeof(Float4));
    }
    else if (ch.typeDelta == 3)//Rotation
    {
        Quaternion lr = Quaternion(low.x, low.y, low.z, low.w);
        Quaternion ur = Quaternion(upp.x, upp.y, upp.z, upp.w);
        Quaternion mx = Math::Slerp(lr, ur, tt).normalize();
        Float4 val = Float4(mx.component.m128_f32[0], mx.component.m128_f32[1], mx.component.m128_f32[2], mx.component.m128_f32[3]);
        model.nodes[ch.idxNode].extensions["poserot"] = toVal(&val, sizeof(Float4));
    }
    else if (ch.typeDelta == 2)//Scale
    {
        Float4 val = low * (1.0 - tt) + upp * tt;
        model.nodes[ch.idxNode].extensions["posesca"] = toVal(&val, sizeof(Float4));
    }
}

//付録C：スプライン補間(https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-c-spline-interpolation)
template <typename T> T cubicSpline(float tt, T v0, T bb, T v1, T aa)
{
    const auto t2 = tt * tt;
    const auto t3 = t2 * tt;
    return (2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + tt) * bb + (-2 * t3 + 3 * t2) * v1 + (t3 - t2) * aa;
}

void gltfInterpolateSpline(Channel& ch, int lowframe, int uppframe, float lowtime, float upptime, float tt)
{
    auto delta = upptime - lowtime;
    auto v0 = ch.deltaKeyframes[3 * lowframe + 1].second;
    auto aa = delta * ch.deltaKeyframes[3 * uppframe + 0].second;
    auto bb = delta * ch.deltaKeyframes[3 * lowframe + 2].second;
    auto v1 = ch.deltaKeyframes[3 * uppframe + 1].second;

    //    _RPTN(0, "** low=%d upp=%d lowf=%.04f ippf=%.04f \n", lowframe, uppframe, lowtime, upptime);

    if (ch.typeDelta == 4)
    {
        //TBD weight
    }
    else if (ch.typeDelta == 1) //Translate
    {
        Float4 val = cubicSpline(tt, v0, bb, v1, aa);
        model.nodes[ch.idxNode].extensions["posetra"] = toVal(&val, sizeof(Float4));
    }
    else if (ch.typeDelta == 3) //Rotation
    {
        Float4 val = cubicSpline(tt, v0, bb, v1, aa);
        Quaternion qt = Quaternion(val.x, val.y, val.z, val.w).normalize();
        val = Float4(qt.component.m128_f32[0], qt.component.m128_f32[1], qt.component.m128_f32[2], qt.component.m128_f32[3]);
        model.nodes[ch.idxNode].extensions["poserot"] = toVal(&val, sizeof(Float4));
    }
    else if (ch.typeDelta == 2) //Scale
    {
        Float4 val = cubicSpline(tt, v0, bb, v1, aa);
        model.nodes[ch.idxNode].extensions["posesca"] = toVal(&val, sizeof(Float4));
    }
}

void gltfCalcSkeleton(tinygltf::Node& node, Mat4x4& matparent = Mat4x4().Identity())
{
    Array<float> r = toFloat(node.extensions["poserot"]);
    Array<float> t = toFloat(node.extensions["posetra"]);
    Array<float> s = toFloat(node.extensions["posesca"]);

    Quaternion rr = Quaternion(r[0], r[1], r[2], r[3]);
    Float3     tt = Float3(t[0], t[1], t[2]);
    Float3     ss = Float3(s[0], s[1], s[2]);

    Mat4x4 poserot = rr.toMatrix();
    Mat4x4 posetra = Mat4x4().Identity().Translate(tt);
    Mat4x4 posesca = Mat4x4().Identity().Scale(ss);
    Mat4x4 matpose = posesca * poserot * posetra;

    Mat4x4 matlocal = toMat(node.extensions["matlocal"]);

    if (0 == std::memcmp(&Mat4x4().Identity(), &matpose, sizeof(Mat4x4)))
        matpose = matlocal;

    Mat4x4 mat = matpose * matlocal.inverse();
    Mat4x4 matworld = mat * matlocal * matparent;

    node.extensions["matworld"] = toVal(&matworld, sizeof(Mat4x4));

    for (int cc = 0; cc < node.children.size(); cc++)
        gltfCalcSkeleton(model.nodes[node.children[cc]], matworld);
}

void gltfPrecomputeMesh(int idxframe, PrecomputeAnime& anime, tinygltf::Node& node)
{
    for (int pp = 0; pp < model.meshes[node.mesh].primitives.size(); pp++)
    {
        auto& pr = model.meshes[node.mesh].primitives[pp];
        auto& map = model.accessors[pr.attributes["POSITION"]];

        int opos, otex, onormal, ojoints, oweights, oidx;

        auto& bpos = *getBuffer(model, pr, "POSITION", &opos);
        auto& btex = *getBuffer(model, pr, "TEXCOORD_0", &otex);
        auto& bnormal = *getBuffer(model, pr, "NORMAL", &onormal);
        auto& bjoint = *getBuffer(model, pr, "JOINTS_0", &ojoints);
        auto& bweight = *getBuffer(model, pr, "WEIGHTS_0", &oweights);
        auto& bidx = *getBuffer(model, pr, &oidx);

        Array<MeshVertex> vertices;
        for (int vv = 0; vv < map.count; vv++)
        {
            float* vertex = nullptr, * texcoord = nullptr, * normal = nullptr;
            vertex = (float*)&bpos.data.at(vv * 12 + opos);
            texcoord = (float*)&btex.data.at(vv * 8 + otex);
            normal = (float*)&bnormal.data.at(vv * 12 + onormal);

            MeshVertex mv;
            mv.position = Float3(vertex[0], vertex[1], vertex[2]);
            mv.texcoord = Float2(texcoord[0], texcoord[1]);
            mv.normal = Float3(normal[0], normal[1], normal[2]);

            //CPUスキニング
            if (pr.attributes["JOINTS_0"] && pr.attributes["WEIGHTS_0"])
            {
                Mat4x4 matskin = Mat4x4().Identity();

                auto joint = (uint16_t*)&bjoint.data.at(vv * 8 + ojoints);//1頂点あたり4JOINT
                auto weight = (float*)&bweight.data.at(vv * 16 + oweights);
                Word4  j4 = Word4(joint[0], joint[1], joint[2], joint[3]);
                Float4 w4 = Float4(weight[0], weight[1], weight[2], weight[3]);

                //スケルトン姿勢(接続されるボーンは4つ)
                matskin = w4.x * precAnimes[0].Joints[node.skin][j4.x] +
                          w4.y * precAnimes[0].Joints[node.skin][j4.y] +
                          w4.z * precAnimes[0].Joints[node.skin][j4.z] +
                          w4.w * precAnimes[0].Joints[node.skin][j4.w];

                auto matpos = matskin.transform(Float4(mv.position, 1.0f));
                auto matnor = matskin.inverse().transposed();

                MeshVertex smv;
                smv.position = Float3(matpos.x, matpos.y, matpos.z) / matpos.w;
                smv.normal = matnor.transform(mv.normal);

                if (pr.attributes["TEXCOORD_0"]) smv.texcoord = Float2(texcoord[0], texcoord[1]);
                vertices.emplace_back(smv);
            }
            else //スキニングなしそのまま描画
            {
                vertices.emplace_back(mv);
            }
        }

        MeshData md;
        if (pr.indices > 0)
        {
            auto& mapi = model.accessors[pr.indices];
            Array<uint32_t> indices;
            for (int ii = 0; ii < mapi.count; ii++)
            {
                uint32_t idx;
                idx = (mapi.componentType == 5123) ? *(uint16_t*)&bidx.data.at(ii * 2 + oidx) :   //16bit
                      (mapi.componentType == 5125) ? *(uint32_t*)&bidx.data.at(ii * 4 + oidx) : 0;//32bit

                indices.emplace_back(idx);
            }

            //基準モデルのメッシュを生成
            md = MeshData(vertices, indices);

            vertices.clear();
            indices.clear();
        }

        auto texcol = 0;// テクスチャ頂点色識別子 b0=頂点色 b1=テクスチャ
        Texture tex;
        ColorF col;

        //Meshes->Primitive->Metarial->baseColorTexture.json_double_value.index->images
        if (pr.material >= 0)
        {
            auto& nt = model.materials[pr.material].additionalValues["normalTexture"];	//法線マップ
            int idx = -1;
            auto& mmv = model.materials[pr.material].values;
            auto& bcf = mmv["baseColorFactor"];			                                //色
            if (mmv.count("baseColorTexture"))
                idx = model.textures[(int)mmv["baseColorTexture"].json_double_value["index"]].source;

            //頂点色を登録
            if (bcf.number_array.size()) col = ColorF(bcf.number_array[0], bcf.number_array[1], bcf.number_array[2], bcf.number_array[3]);
            else                         col = ColorF(1, 1, 1, 1);

            //materials->textures->images
            if (idx >= 0 && model.images.size())
            {
                tex = Texture();
                if (model.images[idx].bufferView >= 0)
                {
                    auto& bgfx = model.bufferViews[model.images[idx].bufferView];
                    auto bimg = &model.buffers[bgfx.buffer].data.at(bgfx.byteOffset);

                    ByteArray teximg((void*)bimg, bgfx.byteLength);
                    tex = Texture(std::move(teximg), TextureDesc::For3D);
                }
                else
                {
                    auto& mii = model.images[idx].image;
                    ByteArray teximg((void*)&mii, mii.size());
                    tex = Texture(std::move(teximg), TextureDesc::For3D);
                }
                texcol |= 2;
            }
            texcol |= 1;
        }

        if (idxframe == 0)  //テクスチャと頂点色の登録はフレーム0に保存。他は共通
        {
            anime.meshTexs.emplace_back(tex);       // テクスチャ追加
            anime.meshColors.emplace_back(col);         // 頂点色追加
        }

        anime.Frames[idxframe].Meshes.emplace_back(DynamicMesh(md));    // メッシュ追加
        anime.Frames[idxframe].TexCol.emplace_back(texcol);             // テクスチャ頂点色識別子追加
    }
}

void gltfDrawPrecomputeAnime(PrecomputeAnime& anime, Float3& tra, Float3& sca, Quaternion& rot)
{
    auto& cfr = anime.currentframe;

    auto& frame = anime.Frames[cfr];
    for (auto pp = 0; pp < frame.Meshes.size(); pp++)
    {
        if (frame.TexCol[pp] & 2)      frame.Meshes[pp].rotated(rot).scaled(sca).translated(tra).draw(anime.meshTexs[pp]);
        else if (frame.TexCol[pp] & 1) frame.Meshes[pp].rotated(rot).scaled(sca).translated(tra).draw(anime.meshColors[pp]);
        else                           frame.Meshes[pp].rotated(rot).scaled(sca).translated(tra).draw();
    }
#ifndef STOPMOTION
    if (++cfr >= anime.Frames.size() - 2) cfr = 0;    //Todo:スプラインバグ隠蔽工作-2
#endif
}

void Main()
{
    Profiler::EnableWarning(false); //アセット大量破棄警告を無効

    const Font fontS(10);           //フォントサイズ指定
    Window::Resize(800, 450);       //画面サイズ
    auto ww = Window::Width();
    auto hh = Window::Height();
    Graphics::SetBackground(Palette::Dimgray);

    tinygltf::TinyGLTF loader;      //Siv3Dkunモデルを読み込み
    std::string err, warn;
    if (!loader.LoadBinaryFromFile(&model, &err, &warn, "Siv3Dkun.glb"))
        loader.LoadBinaryFromFile(&model, &err, &warn, "../Siv3Dkun.glb");

    gltfSetupModel( model, 60 );    //glTFモデル準備、60フレーム

    Stopwatch stopwatch;
    stopwatch.start();               //ストップウォッチ開始

    while (System::Update())
    {
        Graphics3D::FreeCamera();

#ifdef STOPMOTION
        if (AnyKeyClicked() && GetChars().substr(0, 1) == L" " )  //キー押下(シフト以外)
        {
            precAnimes[0].currentframe += 1;
            if (precAnimes[0].currentframe == precAnimes[0].Frames.size())precAnimes[0].currentframe = 0;
        }
#endif

        auto min = stopwatch.min();
        auto sec = stopwatch.s() - min * 60;
        auto ms = stopwatch.ms() - stopwatch.s() * 1000;

        String t1 = L"#1分:#2秒:#3ms経過";
        fontS(t1.replace(L"#1", Pad(min, { 3,L'0' }))
                .replace(L"#2", Pad(sec, { 2,L'0' }))
                .replace(L"#3", Pad(ms, { 3,L'0' }))).draw(ww - 200, hh - 55);

        auto ms1 = stopwatch.ms();

        gltfDrawPrecomputeAnime(precAnimes[0], Float3(0, 0, 0), Float3(5, 5, 5), Quaternion().rollPitchYaw(0, 0, 3.14 / 2));

        auto ms2 = stopwatch.ms();

        String t2 = L"Render:#1ms";
        fontS(t2.replace(L"#1", Pad(ms2 - ms1, { 5,L' ' }))).draw(ww - 200, hh - 35);
    }
}
