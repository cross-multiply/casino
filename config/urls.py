from django.conf import settings
from django.urls import include, path
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.views.generic import TemplateView, RedirectView
from django.views import defaults as default_views

from chipcounting.api.views import DeviceDataUpload, RFIDEventsUpload, \
    DownloadImageZip, DevicePing, RunCVCodeForUpload, DownloadDebugImageZip


urlpatterns = [
    url(r'^$', RedirectView.as_view(url='http://business.chipcounting'), name='home'),
    #url(r'^about/$', TemplateView.as_view(template_name='pages/about.html'), name='about'),

    # Django Admin, use {% url 'admin:index' %}
    url(settings.ADMIN_URL, admin.site.urls),
    #path('grapelli/', include('grappelli.urls')),
    path('api/device/<int:device_id>/rfid', RFIDEventsUpload.as_view()),
    path('api/device/<int:device_id>/data/<uuid:upload_uuid>', DeviceDataUpload.as_view()),
    path('api/device/<int:device_id>/ping', DevicePing.as_view()),
    path('api/data/raw/<int:row_id>/images', DownloadImageZip.as_view(), name='download-raw-images'),
    path('api/data/debug/<int:row_id>/images', DownloadDebugImageZip.as_view(), name='download-debug-images'),
    path('api/data/raw/<int:row_id>/process', RunCVCodeForUpload.as_view(), name='run-cv-code'),
    url(r'^api-auth/', include('rest_framework.urls')),

    # User management
    url(r'^users/', include('chipcounting.users.urls', namespace='users')),
    url(r'^accounts/', include('allauth.urls')),

    # Your stuff: custom urls includes go here
    url(r'^explorer/', include('explorer.urls')),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

admin.site.site_title = 'ChipCounting'
admin.site.site_header = 'ChipCounting Admin'

if settings.DEBUG:
    # This allows the error pages to be debugged during development, just visit
    # these url in browser to see how these error pages look like.
    urlpatterns += [
        url(r'^400/$', default_views.bad_request, kwargs={'exception': Exception('Bad Request!')}),
        url(r'^403/$', default_views.permission_denied, kwargs={'exception': Exception('Permission Denied')}),
        url(r'^404/$', default_views.page_not_found, kwargs={'exception': Exception('Page not Found')}),
        url(r'^500/$', default_views.server_error),
    ]
    if 'debug_toolbar' in settings.INSTALLED_APPS:
        import debug_toolbar
        urlpatterns = [
            url(r'^__debug__/', include(debug_toolbar.urls)),
        ] + urlpatterns
