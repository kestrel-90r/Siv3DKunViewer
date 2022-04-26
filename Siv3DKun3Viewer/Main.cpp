# include <Siv3D.hpp> // OpenSiv3D v0.6.3
# include <Siv3D/EngineLog.hpp>

# include "PixieMesh.hpp"
# include "PixieCamera.hpp"

constexpr ColorF        BGCOLOR = { 0.8, 0.9, 1.0, 0 };
constexpr TextureFormat TEXFORMAT = TextureFormat::R8G8B8A8_Unorm_SRGB;
constexpr Size			WINDOWSIZE = { 1280, 768 };
constexpr StringView	APATH = U"../Asset/";
constexpr RectF			PIPWINDOW{ 900,512,370,240 };
constexpr ColorF        WHITE = ColorF{ 1,1,1,USE_COLOR };
constexpr Vec2			TREECENTER{ -100, -100 };
constexpr double		TREERASIUS = 100;
constexpr double		VOLALITY = 40;
constexpr double		TONAKAISPEED = 0.00003;
constexpr Rect          RECTVRM = { -300, 200, 1280, 768 };

struct ActorRecord
{
	ColorF Color = WHITE;
	Float3 Pos = Float3{ 0,0,0 };
	Float3 Sca = Float3{ 1,1,1 };
	Float3 rPos = Float3{ 0,0,0 };
	Float3 eRot = Float3{ 0,0,0 };
	Quaternion qRot = Quaternion::Identity();
	Float3 eyePos = Float3{ 0,0,0 };
	Float3 focusPos = Float3{ 0,0,0 };
};

enum SELECTTARGET
{
	ST_S3DKUN, ST_FONT, ST_GND, ST_CAMERA, ST_VRM, 
	NUMST
};

enum SELECTBONE
{
	BONE_UPPERBODY, BONE_UPPERBODY2, BONE_NECK, BONE_HEAD, BONE_EYEL, BONE_EYER, BONE_HAIR1, BONE_HAIR2, BONE_HAIR3,
	BONE_SHOULDERL, BONE_SHOULDERR, BONE_ARML, BONE_ARMR, BONE_ELBOWL, BONE_ELBOWR, BONE_WRISTL, BONE_WRISTR,
	BONE_LOWERBODY, BONE_LEGL, BONE_LEGR, BONE_KNEEL, BONE_KNEER, BONE_ANKLEL, BONE_ANKLER,
	BONE_TAIL1, BONE_TAIL2, BONE_TAIL3, BONE_TAIL4, BONE_TAIL5, BONE_TAIL6,
	BONE_LEASH1, BONE_LEASH2,
	NUMBONE
};

enum SELECTMORPH
{
	VRC_BLINKL, VRC_BLINKR,	VRC_LOWLIDL, VRC_LOWLIDR,
	VRC_SIL, VRC_AA, VRC_CH, VRC_IH, VRC_DD, VRC_NN, VRC_KK, VRC_EE, VRC_SS, VRC_FF, VRC_OH, VRC_OU, VRC_PP, VRC_RR, VRC_TH,
	VRM_NEUTRAL, VRM_A, VRM_I, VRM_U, VRM_E, VRM_O, VRM_BLINK,
	OPENMOUTH, OPENMOUTH2, OPENMOUTH3, CLOSEMOUTH, CLOSEMOUTH2, CLOSEMOUTH3, SMILEMOUTH, CLOSEEYES,CLOSEEYES2,SMALLEYES,SMILEEYES,SMILEEYES2,
	SQUAREEYES, KIRAKIRAEYES, NAMIDAEYES, NIRAMIEYES, LOWEREYEBROW, HIGHEREYEBROW, GAMANEYEBROW, KOMARUEYEBROW, SMILEEYEBROW, STANDBY, SMILE,
	NANN, EKKA, SUPPA, NIYARI, NONENOSE, NONEJAW,
	NUMMORPH
};

Array<int32> handleBone( NUMBONE, -1);

Array<ActorRecord> actorRecords;

Array<PixieMesh> pixieMeshes(NUMST);
PixieCamera cameraMain(WINDOWSIZE);

void updateCamera(const Vec3& focuspos, PixieMesh& mesh)
{
	mesh.camera.setFocusPosition(focuspos);

	float speed = 0.01f;
	if (KeyRControl.pressed()) speed *= 5;
	if (KeyLeft.pressed())	mesh.rPos = mesh.camera.arcX(-speed).getEyePosition();
	if (KeyRight.pressed()) mesh.rPos = mesh.camera.arcX(+speed).getEyePosition();
	if (KeyUp.pressed())	mesh.rPos = mesh.camera.arcY(-speed).getEyePosition();
	if (KeyDown.pressed())	mesh.rPos = mesh.camera.arcY(+speed).getEyePosition();

	mesh.camera.updateView();
	mesh.camera.updateViewProj();
	mesh.setRotateQ(mesh.camera.getQForward());
}

void updateMainCamera(const PixieMesh& model, PixieCamera& camera)
{
	Float3 eyepos = camera.getEyePosition();
	Float3 focuspos = camera.getFocusPosition();

	Float2 delta = Cursor::DeltaF();
	Float3 vector = eyepos.xyz() - focuspos;
	Float2 point2D = Cursor::PosF();
	Float3 distance;

	float speedM = 1;
	if (KeyLControl.down() || MouseM.down())								//中ボタンドラッグ：回転
	{
		const Ray mouseRay = camera.screenToRay(Cursor::PosF());
		if (const auto depth = mouseRay.intersects(model.ob))
		{
			Float3 point3D = mouseRay.point_at(*depth);						//ポイントした3D座標を基点
			distance = point3D - eyepos;

			Float3 identity = distance.normalized();
			speedM = distance.length() / identity.length() / 100;
		}
	}

	if (KeyLControl.pressed() || MouseM.pressed())
	{
		bool rev = camera.dolly((float)Mouse::Wheel() * speedM / 5, true);	//中ボタンウィール：拡縮
		if (rev)
		{
			rev = camera.setEyePosition(eyepos).dolly((float)Mouse::Wheel() * speedM / 100, true);
			if (rev)
			{
				rev = camera.setEyePosition(eyepos).dolly((float)Mouse::Wheel() * speedM / 1000, true);
				if (rev) camera.setEyePosition(eyepos);
			}
		}
	}

	if (MouseR.pressed())							//右ボタンドラッグ：平行移動
	{
		if (KeyLShift.pressed()) speedM *= 5;		//Shift押下で5倍速
		camera.trackX(delta.x * speedM / 10);
		camera.craneY(delta.y * speedM / 10);
	}

	Mat4x4 mrot = Mat4x4::Identity();
	if (MouseM.pressed())
	{
		Float4 ve = { 0,0,0,0 };  // 視点移動量
		Float3 vf = { 0,0,0 };    // 注視点移動量
		if (KeyLShift.pressed())  // Shiftで5倍速
		{
			speedM = 5;
			ve *= speedM;
			vf *= speedM;
		}
		camera.arcX(delta.x * speedM / 100);
		camera.arcY(delta.y * speedM / 100);
		camera.setUpDirection(Float3{ 0,1,0 });
	}

	float speedK = camera.getBasisSpeed() / 100 * 5;
	if (KeyLShift.pressed()) speedK *= 10;			//Shift押下で5倍速

	if (KeyW.pressed()) camera.dolly(+speedK, true);
	if (KeyA.pressed()) camera.trackX(+speedK);
	if (KeyS.pressed()) camera.dolly(-speedK, true);
	if (KeyD.pressed()) camera.trackX(-speedK);

	if (KeyE.pressed()) camera.craneY(+speedK);
	if (KeyX.pressed()) camera.craneY(-speedK);
	if (KeyQ.pressed()) camera.tilt(+speedK / 100);
	if (KeyZ.pressed()) camera.tilt(-speedK / 100);

	camera.setUpDirection(Float3{ 0,1,0 });
}




template <typename V, typename T>
V guiValue(const Font& font, const String& text, Rect rect, float wheel, V initvalue, V value, T min, T max)
{
	if (rect.mouseOver() && wheel != 0) value = Math::Clamp(value + wheel, min, max);
	if (rect.leftClicked()) value = initvalue;
	font(text).draw(rect, rect.mouseOver() ? ColorF(1) : ColorF(0));
	return value;
}


void Main()
{
	const Font fontS(15);

	//ウィンドウ初期化
	Window::Resize(WINDOWSIZE);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetBackground(BGCOLOR);

	//描画レイヤ(レンダーテクスチャ)初期化
	static MSRenderTexture rtexMain = { (unsigned)WINDOWSIZE.x, (unsigned)WINDOWSIZE.y, TEXFORMAT, HasDepth::Yes };
	static MSRenderTexture rtexSub = { (unsigned)WINDOWSIZE.x, (unsigned)WINDOWSIZE.y, TEXFORMAT, HasDepth::Yes };

	//メッシュ設定
	const Float3 SCALE_111 = Float3{ 1,1,1 };	//拡縮率
	const Float3 ROT_000 = Float3{ 0,0,0 };		//回転(オイラー角)
	const Float3 RPOS_000 = Float3{ 0,0,0 };	//座標からの相対座標
	const Float3 SPIN_PITCH = Float3{ 0,1,0 };	//スピン(四元数RPY回転)

	PixieMesh& meshSiv3Dkun = pixieMeshes[ST_S3DKUN] = PixieMesh{ APATH + U"Siv3DKun3.019.glb", Float3{0, -15, 0}, Float3{ 10, 10, 10 }, ROT_000, ROT_000, RPOS_000, SPIN_PITCH };
	PixieMesh& meshGND = pixieMeshes[ST_GND]		 = PixieMesh{ APATH + U"Ground.glb",    Float3{0, 0, 0} };
	PixieMesh& meshCamera = pixieMeshes[ST_CAMERA] = PixieMesh{ APATH + U"Camera.glb",      Float3{0, 20, 40},  SCALE_111, ROT_000, ROT_000, RPOS_000, SPIN_PITCH };
	PixieMesh& meshVRM = pixieMeshes[ST_VRM]       = PixieMesh{ APATH + U"Siv3DKun3.019.glb",   Float3{ 0, -50, 0 }, Float3{ 20, 20, 20 }, Float3{ 0, 0, 0 }, Float3{ 0, 0, 0 } };

	//メッシュ初期化

//アニメ+モーフ
//	meshSiv3Dkun.initModel(MODELANI, WINDOWSIZE, NOTUSE_STRING, USE_MORPH, nullptr, HIDDEN_BOUNDBOX, 240, 1);
//	meshSiv3Dkun.initModel(MODELNOA, WINDOWSIZE, NOTUSE_STRING, USE_MORPH);

	meshCamera.initModel(MODELNOA, WINDOWSIZE);
	meshVRM.initModel(MODELVRM, WINDOWSIZE, NOTUSE_STRING, USE_MORPH);

//	meshFont.initModel(MODELNOA, WINDOWSIZE, USE_STRING, NOTUSE_MORPH);
	meshGND.initModel(MODELNOA, WINDOWSIZE);

	//可動ボーン登録
	handleBone[BONE_NECK] = meshVRM.gltfGetJoint(U"neck");
	handleBone[BONE_EYEL] = meshVRM.gltfGetJoint(U"eye_L");
	handleBone[BONE_EYER] = meshVRM.gltfGetJoint(U"eye_R");

	//メインカメラ初期化
	Float4 eyePosMain = { 0, 1, 10.001, 0 };		//視点 XYZは座標、Wはロール用。オイラー角で保持
	Float3 focusPosMain = { 0,  0, 0 };

	cameraMain = PixieCamera(WINDOWSIZE, 45_deg, eyePosMain.xyz(), focusPosMain, 0.05);
	meshCamera.camera = PixieCamera(WINDOWSIZE, 45_deg, meshCamera.Pos, meshVRM.Pos , 0.05);

	double progressPos = 0;	//現在位置をスタート位置に設定
	while (System::Update())
	{
		//メインレイヤ描画
		{
			const ScopedRenderTarget3D rtmain{ rtexMain.clear(BGCOLOR) };

			Graphics3D::SetCameraTransform(cameraMain.getViewProj(), cameraMain.getEyePosition());
			Graphics3D::SetGlobalAmbientColor(ColorF{ 1.0 });
			Graphics3D::SetSunColor(ColorF{ 1.0 });
			{
				const ScopedRenderStates3D rs{ SamplerState::RepeatAniso, RasterizerState::SolidCullFront };

				//制御
				updateMainCamera(meshGND, cameraMain);
				updateCamera(meshVRM.Pos + Float3{ 0,50,0 }, meshCamera);

				//描画
				meshGND.drawMesh();
			}

			Graphics3D::Flush();
			rtexMain.resolve();
			Shader::LinearToScreen(rtexMain);

#define USE_VRMCAMERA
#ifdef USE_VRMCAMERA
			//サブレイヤ描画
			{
				//透過ブレンド
				BlendState blend = BlendState::Default3D;
				blend.srcAlpha = Blend::SrcAlpha;
				blend.dstAlpha = Blend::DestAlpha;
				blend.opAlpha = BlendOp::Max;
//				const ScopedRenderStates3D rs{ blend, SamplerState::RepeatAniso, RasterizerState::SolidCullFront };

				const ScopedRenderStates3D bs{ BlendState::OpaqueAlphaToCoverage , SamplerState::RepeatAniso, RasterizerState::SolidCullFront };
				const ScopedRenderTarget3D rtsub{ rtexSub.clear(ColorF{ 1,1,1,0 }) };

				Graphics3D::SetCameraTransform( meshCamera.camera.getViewProj(),
					                            meshCamera.camera.getEyePosition());


				meshVRM.drawVRM();				// VRM描画

//アニメ+モーフテスト
//				meshSiv3Dkun.drawMesh();
//				meshSiv3Dkun.drawAnime(1).nextFrame(1);

				if (RECTVRM.mouseOver())
				{
					if (MouseL.pressed()) //左ボタンで首ボーン操作
					{
						static Float3 rotN = {0,0,0};
						static double Rrot, Lrot, Rpos, Lpos;

						double xx = ToRadians(Cursor::DeltaF().x * +10);
						double yy = ToRadians(Cursor::DeltaF().y * -10);

						rotN += Float3{ yy,xx,0 };
						rotN.x = Clamp(rotN.x, -20.0f, +20.0f);
						rotN.y = Clamp(rotN.y, -50.0f, +50.0f);
						meshVRM.gltfSetJoint(handleBone[BONE_NECK], {0,0,0}, rotN);

						double vy = -Cursor::DeltaF().y / 1000;
						xx /= 5;
						Rrot -= xx;
						Lrot -= xx;
						meshVRM.gltfSetJoint(handleBone[BONE_EYEL], { 0, 0, 0}, { 0, Clamp(Lrot,-3.0,12.0),0 });
						meshVRM.gltfSetJoint(handleBone[BONE_EYER], { 0, 0, 0}, { 0, Clamp(Rrot,-12.0,3.0),0 });
					}
				}

				Graphics3D::Flush();
				rtexSub.resolve();
				Shader::LinearToScreen(rtexSub, RECTVRM.pos);
			}
#endif

#define USE_DEBUGGUI
#ifdef USE_DEBUGGUI
			{

//アニメ+モーフテスト
//				auto& mt = meshSiv3Dkun.morphTarget;

				auto& mt = meshVRM.morphTarget;
				double wheel = Mouse::Wheel()/10;
				uint32 i = 0;
				Rect base = Rect{ 1000,100,150,22 };
				mt[i++].Weight = guiValue(fontS, U"{}.ウィンクL:{:.3f}"_fmt(i,mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.ウィンクR:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.半目L:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.半目R:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);

				mt[i++].Weight = guiValue(fontS, U"{}.VRC待機:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_AA:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_CH:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_IH:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_DD:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_NN:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_KK:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_EE:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_SS:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_FF:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_OH:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_OU:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_PP:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_RR:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRC_TH:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM待機:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_あ:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_い:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_う:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_え:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_お:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.VRM_瞬:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);

				base = Rect{ 1150,100,150,22 };
				mt[i++].Weight = guiValue(fontS, U"{}.開口1:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.開口2:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.開口3:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.閉口1:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.閉口2:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.閉口3:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.笑口:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.閉目1:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.閉目2:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.小目:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.笑目1:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.笑目2:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.角目:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.☆目:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.涙目:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.睨目:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.下眉:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.上眉:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.耐眉:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.困眉:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.笑眉:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.待機:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.笑顔:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.不満:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.反論:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.酸味:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.にやり:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.鼻無:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
				mt[i++].Weight = guiValue(fontS, U"{}.顎無:{:.3f}"_fmt(i, mt[i].Weight), base.moveBy(0, 20), wheel, 0.0f, mt[i].Weight, 0.0f, 1.0f);
			}
#endif

		}
	}
}

