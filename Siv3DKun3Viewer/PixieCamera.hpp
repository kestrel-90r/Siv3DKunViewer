//-----------------------------------------------
//
//	This file is part of the Siv3D Engine.
//
//	Copyright (c) 2008-2021 Ryo Suzuki
//	Copyright (c) 2016-2021 OpenSiv3D Project
//
//	Licensed under the MIT License.
//
//-----------------------------------------------

# pragma once

namespace s3d
{
	class alignas(16) PixieCamera //:public BasicCamera3D
	{
//	using BasicCamera3D::BasicCamera3D;

	public:
// from BasicCamera3D
		static constexpr double DefaultVerticalFOV = 30_deg;
		static constexpr double DefaultNearClip = 0.2;

		virtual ~PixieCamera() = default;
		PixieCamera() = default;
		PixieCamera(const PixieCamera&) = default;
		explicit PixieCamera(const Size& sceneSize, double verticalFOV = DefaultVerticalFOV,
							  const Vec3& eyePosition = Vec3{ 0, 4, -4 },
							  const Vec3& focusPosition = Vec3{ 0, 0, 0 },
							  double nearClip = DefaultNearClip,
							  const Vec3& upDirection = Vec3{ 0, 1, 0 }) noexcept
			: m_sceneSize{ sceneSize }
			, m_verticalFOV{ verticalFOV }
			, m_nearClip{ nearClip }
			, m_eyePos{ eyePosition }
			, m_focusPos{ focusPosition }
			, m_upDir{ upDirection }
		{
			updateProj();
			updateView();
			updateViewProj();
		}

		void setSceneSize(const Size& sceneSize) noexcept
		{
			m_sceneSize = sceneSize;
			updateProj();
			updateViewProj();
		}

		inline const Size& getSceneSize() const noexcept
		{
			return m_sceneSize;
		}

		void setProjection(const Size& sceneSize, double verticalFOV, double nearClip = DefaultNearClip) noexcept
		{
			m_sceneSize = sceneSize;
			m_verticalFOV = verticalFOV;
			m_nearClip = nearClip;

			updateProj();
			updateViewProj();
		}

		inline const Mat4x4& SIV3D_VECTOR_CALL getProj() const noexcept
		{
			return m_proj;
		}

		void setView(const Vec3& eyePosition, const Vec3& focusPosition, const Vec3& upDirection = Vec3{ 0, 1, 0 }) noexcept
		{
			m_eyePos = eyePosition;
			m_focusPos = focusPosition;
			m_upDir = upDirection;

			updateView();
			updateViewProj();
		}

		inline const Mat4x4& SIV3D_VECTOR_CALL getView() const noexcept
		{
			return m_view;
		}

		void setUpDirection(const Vec3& upDirection) noexcept
		{
			m_upDir = upDirection;
			updateView();
			updateViewProj();
		}
		inline const Vec3& getUpDirection() const noexcept
		{
			return m_upDir;
		}

		inline double getVerticlaFOV() const noexcept
		{
			return m_verticalFOV;
		}

		inline double getNearClip() const noexcept
		{
			return m_nearClip;
		}

		inline const Vec3& getEyePosition() const noexcept
		{
			return m_eyePos;
		}

		inline const Vec3& getFocusPosition() const noexcept
		{
			return m_focusPos;
		}

		Float3 worldToScreenPoint(const Float3& pos) const noexcept
		{
			Float3 v = SIMD_Float4{ DirectX::XMVector3TransformCoord(
				                    SIMD_Float4{ pos, 0.0f }, m_viewProj) }.xyz();
			v.x += 1.0f;
			v.y += 1.0f;
			v.x *= 0.5f * m_sceneSize.x;
			v.y *= 0.5f;
			v.y = 1.0f - v.y;
			v.y *= m_sceneSize.y;
			return v;
		}

		Float3 screenToWorldPoint(const Float2& pos, const float depth) const noexcept
		{
			Float3 v{ pos, depth };
			v.x /= (m_sceneSize.x * 0.5f);
			v.y /= (m_sceneSize.y * 0.5f);
			v.x -= 1.0f;
			v.y -= 1.0f;
			v.y *= -1.0f;

			const SIMD_Float4 worldPos = DirectX::XMVector3TransformCoord(
				                         SIMD_Float4{ v, 0.0f }, m_invViewProj);
			return worldPos.xyz();
		}

		Ray screenToRay(const Vec2& pos) const noexcept
		{
			const Vec3 rayEnd = screenToWorldPoint(pos, 0.1f);
			return Ray{ m_eyePos, (rayEnd - m_eyePos).normalized() };
		}

		inline Vec3 getLookAtVector() const noexcept
		{
			return (m_focusPos - m_eyePos).normalized();
		}

		Quaternion getLookAtOrientation() const noexcept
		{
			return Quaternion::FromUnitVectorPairs( { Vec3::Forward(), Vec3::Up() },
													{ getLookAtVector(), Vec3::Up() });
		}

		inline const Mat4x4& SIV3D_VECTOR_CALL getInvView() const noexcept
		{
			return m_invView;
		}

		inline const Mat4x4& SIV3D_VECTOR_CALL getViewProj() const noexcept
		{
			return m_viewProj;
		}

		const Mat4x4& SIV3D_VECTOR_CALL getInvViewProj() const noexcept
		{
			return m_invViewProj;
		}

		inline Mat4x4 billboard(const Float3 pos, const Float2 scale) const noexcept
		{
			Mat4x4 m = m_invView;
			m.value.r[0] = DirectX::XMVectorScale(m.value.r[0], scale.x);
			m.value.r[1] = DirectX::XMVectorScale(m.value.r[1], 1.0f);
			m.value.r[2] = DirectX::XMVectorScale(m.value.r[2], scale.y);
			m.value.r[3] = DirectX::XMVectorSet(pos.x, pos.y, pos.z, 1.0f);
			return m;
		}

		void updateProj() noexcept
		{
			const double g = (1.0 / std::tan(m_verticalFOV * 0.5));
			const double s = (static_cast<double>(m_sceneSize.x) / m_sceneSize.y);
			constexpr float e = 0.000001f;

			m_proj = Mat4x4{
				static_cast<float>(g / s), 0.0f, 0.0f, 0.0f,
				0.0f, static_cast<float>(g), 0.0f, 0.0f,
				0.0f, 0.0f, e, 1.0f,
				0.0f, 0.0f, static_cast<float>(m_nearClip * (1.0 - e)), 0.0f
			};
		}

		void updateView() noexcept
		{
			const SIMD_Float4 eyePosition{ m_eyePos, 0.0f };
			const SIMD_Float4 focusPosition{ m_focusPos, 0.0f };
			const SIMD_Float4 upDirection{ m_upDir, 0.0f };
			m_view = DirectX::XMMatrixLookAtLH(eyePosition, focusPosition, upDirection);
			m_invView = m_view.inverse();
		}

		void updateViewProj() noexcept
		{
			m_viewProj = (m_view * m_proj);
			m_invViewProj = m_viewProj.inverse();
		}

		Mat4x4 m_proj = Mat4x4::Identity();
		Mat4x4 m_view = Mat4x4::Identity();
		Mat4x4 m_invView = Mat4x4::Identity();
		Mat4x4 m_viewProj = Mat4x4::Identity();
		Mat4x4 m_invViewProj = Mat4x4::Identity();

		Size m_sceneSize = Scene::DefaultSceneSize;
		double m_verticalFOV = DefaultVerticalFOV;
		double m_nearClip = DefaultNearClip;

		Vec3 m_eyePos = Vec3{ 0, 4, -4 };
		Vec3 m_focusPos = Vec3{ 0, 0, 0 };
		Vec3 m_upDir = Vec3{ 0, 1, 0 };
		Vec3 m_eyePosDelta = Vec3{ 0, 0, 0 };
		Vec3 m_focusPosDelta = Vec3{ 0, 0, 0 };


// itakawa added camerawork features

		PixieCamera& setEyePosition(const Vec3& eyepos) noexcept
		{
			m_eyePos = eyepos;
			return *this;
		}

		PixieCamera& setFocusPosition(const Vec3& focuspos) noexcept
		{
			m_focusPos = focuspos;
			return *this;
		}

		inline const Vec3& getEyePosDelta() const noexcept
		{
			return m_eyePosDelta;
		}

		inline const Vec3& getFocusPosDelta() const noexcept
		{
			return m_focusPosDelta;
		}

		float getBasisSpeed()
		{
			Float3 distance = m_focusPos - m_eyePos;
			Float3 identity = distance.normalized();
			return distance.length() / identity.length() ;
		}

		bool dolly(float forwardback, bool movefocus )
		{
			Float3 dirB = (m_focusPos - m_eyePos).normalized();
			m_eyePosDelta = forwardback * dirB;
			m_focusPosDelta = m_eyePosDelta;

			m_eyePos += m_eyePosDelta;
			if (movefocus) m_focusPos = m_eyePos + dirB;
			Float3 dirA = (m_focusPos - m_eyePos);

			return ( Math::Sign(dirA.x) != Math::Sign(dirB.x) &&//目標点に達したら真を返す（止める等の処置をする）
					 Math::Sign(dirA.y) != Math::Sign(dirB.y) &&
					 Math::Sign(dirA.z) != Math::Sign(dirB.z));
		}

		PixieCamera& dolly( float forwardback )
		{
			Float3 dir = (m_focusPos - m_eyePos).normalized();
			m_eyePosDelta = forwardback * dir;
			m_focusPosDelta = m_eyePosDelta;
			m_eyePos += m_eyePosDelta;
			m_focusPos = m_eyePos + dir;
			return *this;
		}

		PixieCamera& panX(float leftright)
		{
			Float3 forward = m_focusPos - m_eyePos;
			Float3 dir = forward.normalized();
			Float3 right = dir.cross(m_upDir);
			m_upDir = right.cross(dir).normalized();

			SIMD_Float4 qrot = DirectX::XMQuaternionRotationAxis(SIMD_Float4{ m_upDir, 0 }, leftright);
			SIMD_Float4 focus = DirectX::XMVector3Rotate(SIMD_Float4{ forward, 0 }, qrot);

			Float3 oldfocus = m_focusPos;
			m_eyePosDelta = Float3{0,0,0};
			m_focusPos = m_eyePos + focus.xyz() ;
			m_focusPosDelta = m_focusPos - oldfocus;
			return *this;
		}

		PixieCamera& panY(float updown)
		{
			Float3 forward = m_focusPos - m_eyePos;
			Float3 dir = forward.normalized();
			Float3 right = dir.cross(m_upDir);
			m_upDir = right.cross(forward).normalized();

			SIMD_Float4 qrot = DirectX::XMQuaternionRotationAxis(SIMD_Float4{ right, 0 }, updown);
			SIMD_Float4 focus = DirectX::XMVector3Rotate(SIMD_Float4{ forward, 0 }, qrot);

			Float3 oldfocus = m_focusPos;
			m_eyePosDelta = Float3{ 0,0,0 };
			m_focusPos = m_eyePos + focus.xyz() ;
			m_focusPosDelta = m_focusPos - oldfocus;
			return *this;
		}

		PixieCamera& tilt(float updown)
		{
			panY(updown);
			return *this;
		}

		PixieCamera& trackX(float leftright)
		{
			Float3 fwd = m_focusPos - m_eyePos;
			Float3 dir = fwd.normalized();
			Float3 right = dir.cross(m_upDir).normalized();

			dir = leftright * right;
			m_eyePosDelta = dir;
			m_focusPosDelta = dir;
			m_eyePos += dir;
			m_focusPos += dir;
			return *this;
		}

		PixieCamera& craneY(float updown)
		{
			Float3 fwd = m_focusPos - m_eyePos;
			Float3 dir = fwd.normalized();
			Float3 right = dir.cross(m_upDir);
			m_upDir = right.cross(dir).normalized();

			dir = updown * m_upDir;
			m_eyePosDelta = dir;
			m_focusPosDelta = dir;
			m_eyePos += dir;
			m_focusPos += dir;
			return *this;
		}

		PixieCamera& arcX(float leftright)
		{
			Float3 forward = m_eyePos - m_focusPos;
			Float3 dir = (-forward).normalized();
			Float3 right = m_upDir.cross(dir);
			m_upDir = dir.cross(right).normalized();

			SIMD_Float4 qrot = DirectX::XMQuaternionRotationAxis(SIMD_Float4{ m_upDir, 0 }, leftright);
			SIMD_Float4 eye = DirectX::XMVector3Rotate(SIMD_Float4{ forward, 0 }, qrot);

			Float3 oldeye = m_eyePos;
			m_eyePos = eye.xyz() + m_focusPos;
			m_eyePosDelta = m_eyePos - oldeye;
			m_focusPosDelta = Float3{ 0,0,0 };
			return *this;
		}

		PixieCamera& arcY(float updown)
		{
			Float3 forward = m_eyePos - m_focusPos;
			Float3 dir = (-forward).normalized();
			Float3 right = m_upDir.cross(dir);
			m_upDir = dir.cross(right).normalized();

			SIMD_Float4 qrot = DirectX::XMQuaternionRotationAxis(SIMD_Float4{ right, 0 }, updown);
			SIMD_Float4 eye = DirectX::XMVector3Rotate(SIMD_Float4{ forward, 0 }, qrot);
			Float3 oldeye = m_eyePos;
			m_eyePos = eye.xyz() + m_focusPos;
			m_eyePosDelta = m_eyePos - oldeye;
			m_focusPosDelta = Float3{ 0,0,0 };
			return *this;
		}

		Quaternion getQLookAt(const Float3& dst, const Float3& src,
						      Float3* up = nullptr, Float3* right = nullptr) const//ポインタを指定する場合は結果を引数で取得
		{
			if (dst == src) return Quaternion::Identity();
			Float3 _up = Float3{ 0,1,0 };
			if (up != nullptr) _up = *up;

			Float3 _forward = dst - src;
			Float3 _dir = _forward.normalized();
			if (_dir == _up) _dir += Float3{ 0.000001,0,0 };	//上と方向が同じ
			Float3 _right = _dir.cross(_up).normalized();
			_up = _dir.cross(_right).normalized();
			if (up != nullptr)    *up = _up ;
			if (right != nullptr) *right = _right;

			return Quaternion::FromUnitVectorPairs( { Float3{0,0,1}, Float3{0,1,0} },
				                                    { -_dir, -_up });
		}

		Quaternion getQForward() const
		{
			return getQLookAt(m_focusPos, m_eyePos );
		}

		Quaternion getQUp() const
		{
			return getQLookAt(m_upDir, m_eyePos );
		}

		Quaternion getQRight() const
		{
			Float3 forward = (m_focusPos - m_eyePos).normalized();
			Float3 right = forward.cross(m_upDir).normalized();
			return getQLookAt(right, m_eyePos);
		}

		Mat4x4 getMatInfront(Float3 pos, Float2 scale = Float2{1,1}, float distance = 5.0f) const
		{
			Float3 infront = ( m_eyePos - pos).normalized();
			pos = infront * distance;
			return billboard( pos, scale);
		}

	};
}
